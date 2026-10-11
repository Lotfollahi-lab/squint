from typing import Callable, Dict, Optional, List, Literal, Union

import scanpy as sc
import anndata as ad

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from ..preprocessors.normalizers import normalize_by_read_depth, normalize_total_log1p


# Allowed values of the opt-in inductive held-out-region graph knobs
# (`SpatialBatchSplit.holdout_graph` / `holdout_test_inedges`).
HOLDOUT_GRAPH_MODES = ('transductive', 'drop', 'rebuild')
HOLDOUT_TEST_INEDGES = ('full', 'test_only', 'none')


def _blob_knn_adjacency(xy, k: int):
    """
    Symmetrised kNN adjacency over `xy` (numpy array, shape (n, 2)), built
    exactly like the dataset blob builds its `spatial_n_neighs_<k>` graphs
    (in_memory_dataset_blob.py, "Build Neighborhood Graphs" and "Build Edge
    Indices"): squint's `spatial_neighbors` wrapper around squidpy with
    coord_type='generic', delaunay=False, radius=None and
    include_self_loop=True (the settings of every blob config in
    examples/run_squint.py), then `A.maximum(A.T)` when A is not symmetric.

    Returns a scipy.sparse matrix of shape (n, n).
    """
    import numpy as np
    import pandas as pd
    from ..preprocessors.graph_constructors import spatial_neighbors

    xy = np.asarray(xy)
    n = int(xy.shape[0])
    k = int(k)
    if n - 1 < k:
        # sklearn's kneighbors() (self excluded) needs k < n.
        print(f"_blob_knn_adjacency: only {n} points, clamping k={k} -> {max(n - 1, 0)}.")
        k = n - 1
    if k < 1:
        import scipy.sparse as sp
        return sp.identity(n, dtype=np.float32, format='csr')

    adata = ad.AnnData(
        obs=pd.DataFrame(index=[str(i) for i in range(n)]),
        obsm={'spatial': xy},
    )
    key_added = 'holdout_rebuild'
    adata = spatial_neighbors(
        adata,
        coord_type='generic',
        spatial_key='spatial',
        delaunay=False,
        n_neighs=k,
        radius=None,
        include_self_loop=True,
        key_added=key_added,
    )
    A = adata.obsp[f"{key_added}_connectivities"]
    if (A != A.T).nnz > 0:
        A = A.maximum(A.T)
    return A


def make_inductive_edge_index(
        edge_index: torch.Tensor,
        test_mask: torch.Tensor,
        xy: torch.Tensor,
        mode: Literal['drop', 'rebuild'],
        k: Optional[int] = None,
        test_inedges: Literal['full', 'test_only', 'none'] = 'full',
        knn_builder: Optional[Callable] = None,
    ) -> torch.Tensor:
    """
    Rewrite the graph of ONE tissue section so that no held-out
    (test_mask) cell can send a message to an observed cell.

    PyG convention: edge_index[0] = source, edge_index[1] = target (messages
    flow src -> dst). NeighborLoader with its default
    subgraph_type='directional' samples the IN-edges of each seed, so after
    this rewrite an observed seed can never pull a held-out node into a
    batch, at any hop depth.

    Parameters:
    ----------
    - edge_index: LongTensor [2, E]
        The section's graph (the blob's symmetrised kNN with self-loops).
    - test_mask: BoolTensor [N]
        True for held-out cells.
    - xy: Tensor [N, 2]
        Coordinates used by mode='rebuild' (the same coordinates the
        held-out boxes were computed on).
    - mode: 'drop' | 'rebuild'
        Observed -> observed edges. 'drop' keeps the original ones (boundary
        observed cells lose their held-out neighbours). 'rebuild' replaces
        them with a symmetrised kNN(k) over observed cells only, built with
        the same constructor, parameters, symmetrisation and self-loop
        convention as the dataset blob (see `_blob_knn_adjacency`).
    - k: int
        Number of neighbours for mode='rebuild'.
    - test_inedges: 'full' | 'test_only' | 'none'
        Edges INTO held-out cells. 'full' keeps all their original in-edges
        (from observed and held-out sources), 'test_only' keeps only those
        from held-out sources, 'none' keeps none (self-loop only).
    - knn_builder: Callable, optional
        Override for the kNN constructor, called as
        `knn_builder(xy_obs_numpy, k)` and returning a scipy.sparse (n_obs,
        n_obs) adjacency. Used by tests that run without squidpy.

    Returns:
    -------
    - LongTensor [2, E']
        Coalesced (sorted by source then target, deduplicated) edge index on
        the input device and dtype. Guarantees: no edge with a held-out
        source and an observed target; every node has exactly one self-loop.

    Notes:
    -----
    - Coordinates must belong to a single section: a 'rebuild' over a
      collated multi-section batch would connect sections.
    """
    if mode not in ('drop', 'rebuild'):
        raise ValueError(f"mode must be 'drop' or 'rebuild', got {mode!r}")
    if test_inedges not in HOLDOUT_TEST_INEDGES:
        raise ValueError(
            f"test_inedges must be one of {HOLDOUT_TEST_INEDGES}, got {test_inedges!r}"
        )
    if mode == 'rebuild' and (k is None or int(k) < 1):
        raise ValueError(f"mode='rebuild' needs a positive k, got k={k!r}")

    device = edge_index.device
    dtype = edge_index.dtype
    t = test_mask.to(device=device, dtype=torch.bool).reshape(-1)
    num_nodes = int(t.numel())
    ei = edge_index.to(dtype=torch.long)
    if num_nodes == 0:
        return edge_index[:, :0].clone()
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(ei.shape)}")
    if ei.numel() > 0 and int(ei.max()) >= num_nodes:
        raise ValueError(
            f"edge_index refers to node {int(ei.max())} but test_mask has only {num_nodes} nodes"
        )
    src, dst = ei[0], ei[1]

    # Edges INTO held-out cells (never reach an observed cell).
    keep_in = t[dst]
    if test_inedges == 'test_only':
        keep_in = keep_in & t[src]
    elif test_inedges == 'none':
        keep_in = keep_in & (src == dst)
    parts = [ei[:, keep_in]]

    # Observed -> observed edges.
    if mode == 'drop':
        parts.append(ei[:, ~t[src] & ~t[dst]])
    else:
        obs_idx = torch.nonzero(~t, as_tuple=False).view(-1)
        if obs_idx.numel() > 0:
            xy_obs = xy.detach()[obs_idx.to(xy.device)].cpu().numpy()
            builder = knn_builder if knn_builder is not None else _blob_knn_adjacency
            A = builder(xy_obs, int(k))
            n_obs = int(obs_idx.numel())
            if tuple(A.shape) != (n_obs, n_obs):
                raise ValueError(
                    f"knn_builder returned shape {tuple(A.shape)}, expected ({n_obs}, {n_obs})"
                )
            # Same conversion as torch_geometric.utils.from_scipy_sparse_matrix
            # (used by the blob): COO row -> source, COO col -> target.
            A = A.tocoo()
            local = torch.stack([
                torch.from_numpy(A.row).to(torch.long),
                torch.from_numpy(A.col).to(torch.long),
            ], dim=0).to(device)
            parts.append(obs_idx[local])

    # Exactly one self-loop per node (loss_utils' 1-hop aggregation and the
    # 'none' in-edge mode rely on it); duplicates are removed below.
    loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    parts.append(torch.stack([loops, loops], dim=0))

    # Coalesce: sort by (source, target) and drop duplicates, the same
    # order as torch_geometric.utils.coalesce(sort_by_row=True).
    out = torch.cat(parts, dim=1)
    key = torch.unique(out[0] * num_nodes + out[1], sorted=True)
    out = torch.stack([
        torch.div(key, num_nodes, rounding_mode='floor'),
        torch.remainder(key, num_nodes),
    ], dim=0)

    n_bad = int((t[out[0]] & ~t[out[1]]).sum())
    if n_bad > 0:
        raise RuntimeError(
            f"make_inductive_edge_index: {n_bad} held-out -> observed edges remain."
        )
    return out.to(dtype=dtype)


def init_gene_count_transforms(
        gene_count_transform_names: List[Literal['SubsetHVG', 'NormalizeFeatures']] = ['SubsetHVG'],
        n_genes: int = 1000,
        norm_method: str = 'total_log1p',
        target_size: int = 10_000,
        apply_CPM: bool = True,
    ) -> List[T.BaseTransform]:
    """
    Initialize a list of PyG transforms to be applied to the raw cell gene counts matrix.

    Parameters:
    ----------
    - gene_count_transform_names: List[Literal['SubsetHVG', 'NormalizeFeatures']]
        The list of PyG transforms to initialize. Currently, only 'SubsetHVG' and 'NormalizeFeatures' are supported.
    - n_genes: int
        The number of highly variable genes to subset the data to.
    - norm_method: str
        The method to use for normalization.
    - target_size: int
        The target size for normalization.
    - apply_CPM: bool
        If True, apply CPM normalization.

    Returns:
    -------
    - List[T.BaseTransform]
        A list of PyG transforms to be applied to data.

    Notes:
    -----
    - Transforms are not order-invariant.
    - If 'SubsetHVG' is listed, it is applied before 'NormalizeFeatures'.
    """
    gene_count_transforms = []

    if 'SubsetHVG' in gene_count_transform_names:
        gene_count_transforms.append(
            SubsetHVG(
                n_genes=n_genes
            )
        )

    if 'NormalizeFeatures' in gene_count_transform_names:
        gene_count_transforms.append(
            NormalizeFeatures(
                norm_method=norm_method,
                target_size=target_size,
                apply_CPM=apply_CPM
            )
        )

    return gene_count_transforms


def init_train_transforms(
        train_transform_names: List[Literal['RandomNodeSplit', 'SpatialNodeSplit', 'SpatialBatchSplit']] = ['RandomNodeSplit'],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        # New parameters for spatial split
        val_region: dict = None,
        test_region: dict = None,
        xy_key: str = 'xy_coordinates',
        # New parameters for batch split
        region: dict = None,
        train_batches: Optional[list] = None,
        val_batches: Optional[list] = None,
        test_batches: Optional[list] = None,
        # Cell-level holdout knobs for SpatialBatchSplit (in-distribution
        # early stopping / best-ckpt selection inside training sections).
        train_val_cell_split: float = 0.0,
        cell_split_seed: int = 0,
        # Per-batch held-out region rectangles for the gene-reconstruction
        # downstream task (SpatialBatchSplit.test_regions).
        test_regions: Optional[Dict[int, Union[dict, List[dict]]]] = None,
        # Opt-in inductive graph for the test_regions sections
        # (SpatialBatchSplit.holdout_graph). Defaults keep the legacy
        # transductive graph.
        holdout_graph: Literal['transductive', 'drop', 'rebuild'] = 'transductive',
        holdout_knn_k: Optional[int] = None,
        holdout_test_inedges: Literal['full', 'test_only', 'none'] = 'full',
    ) -> List[T.BaseTransform]:
    """
    Initialize a list of training-related PyG transforms to be applied to the Data object.

    Parameters:
    ----------
    - train_transform_names: List[Literal['RandomNodeSplit', 'SpatialNodeSplit', 'BatchSplit']]
        The list of PyG transforms to initialize.
    - val_ratio: float
        Fraction of the data to set aside for validation (used with RandomNodeSplit).
    - test_ratio: float
        Fraction of the data to set aside for testing (used with RandomNodeSplit).
    - val_region: dict
        Spatial region for validation (used with SpatialNodeSplit).
    - test_region: dict
        Spatial region for testing (used with SpatialNodeSplit).
    - xy_key: str
        Attribute name for coordinates (used with SpatialNodeSplit).
    - region: dict
        Spatial region for training (used with SpatialBatchSplit).
    - train_batches: list
        List of batch indices for training (used with BatchSplit).
    - val_batches: list
        List of batch indices for validation (used with BatchSplit).
    - test_batches: list
        List of batch indices for testing (used with BatchSplit).
    - holdout_graph, holdout_knn_k, holdout_test_inedges:
        Inductive held-out-region graph knobs (used with SpatialBatchSplit;
        see SpatialBatchSplit for details). Defaults keep the legacy graph.
    """
    train_transforms = []
    
    if 'RandomNodeSplit' in train_transform_names:
        train_transforms.append(
            T.RandomNodeSplit(
                split='train_rest',
                num_val=val_ratio,
                num_test=test_ratio,
                key='y'
            )
        )
    
    if 'SpatialNodeSplit' in train_transform_names:
        train_transforms.append(
            SpatialNodeSplit(
                val_region=val_region,
                test_region=test_region,
                xy_key=xy_key
            )
        )
    
    if 'SpatialBatchSplit' in train_transform_names:
        train_transforms.append(
            SpatialBatchSplit(
                region=region,
                xy_key=xy_key,
                train_batches=train_batches,
                val_batches=val_batches,
                test_batches=test_batches,
                train_val_cell_split=train_val_cell_split,
                cell_split_seed=cell_split_seed,
                test_regions=test_regions,
                holdout_graph=holdout_graph,
                holdout_knn_k=holdout_knn_k,
                holdout_test_inedges=holdout_test_inedges,
            )
        )
    
    return train_transforms


class SpatialNodeSplit(T.BaseTransform):
    def __init__(
        self,
        val_region: dict = None,
        test_region: dict = None,
        xy_key: str = 'xy_coordinates'
    ):
        """
        Split nodes based on spatial regions defined by min/max x and y coordinates.
        All nodes not in val or test regions are automatically assigned to training.
        
        Parameters:
        ----------
        - val_region: dict
            Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max' defining the validation region
        - test_region: dict
            Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max' defining the test region
        - xy_key: str
            The attribute name in the data object containing the xy coordinates
        
        Example:
        --------
        val_region = {'x_min': 100, 'x_max': 150, 'y_min': 0, 'y_max': 100}
        test_region = {'x_min': 150, 'x_max': 200, 'y_min': 0, 'y_max': 100}
        # All other nodes will be in training
        """
        self.val_region = val_region
        self.test_region = test_region
        self.xy_key = xy_key
        
    def _is_in_region(self, coords, region):
        """Check if coordinates are within the specified region."""
        if region is None:
            return torch.zeros(coords.shape[0], dtype=torch.bool)
            
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        x_mask = (x_coords >= region['x_min']) & (x_coords <= region['x_max'])
        y_mask = (y_coords >= region['y_min']) & (y_coords <= region['y_max'])
        
        return x_mask & y_mask
    
    def forward(self, data: Data) -> Data:
        """Apply spatial split to the data object."""
        # Get coordinates
        if not hasattr(data, self.xy_key):
            raise ValueError(f"Data object does not have attribute '{self.xy_key}'")
            
        coords = getattr(data, self.xy_key)
        num_nodes = coords.shape[0]
        
        # Initialize masks
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Apply regional splits for val and test
        if self.val_region is not None:
            val_mask = self._is_in_region(coords, self.val_region)
            
        if self.test_region is not None:
            test_mask = self._is_in_region(coords, self.test_region)
        
        # Handle overlapping regions (test has priority over val)
        val_mask = val_mask & ~test_mask
        
        # All remaining nodes go to training
        train_mask = ~(val_mask | test_mask)
        
        # Assign masks to data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        print(f"SpatialNodeSplit: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
        print(f"Total nodes assigned: {(train_mask | val_mask | test_mask).sum()}/{num_nodes}")
        
        return data


class SpatialBatchSplit(T.BaseTransform):
    def __init__(
            self,
            region: Optional[Union[dict, List[dict]]] = None,
            xy_key: str = 'xy_coordinates',
            train_batches: Optional[list] = None,
            val_batches: Optional[list] = None,
            test_batches: Optional[list] = None,
            train_val_cell_split: float = 0.0,
            cell_split_seed: int = 0,
            test_regions: Optional[Dict[int, Union[dict, List[dict]]]] = None,
            holdout_graph: Literal['transductive', 'drop', 'rebuild'] = 'transductive',
            holdout_knn_k: Optional[int] = None,
            holdout_test_inedges: Literal['full', 'test_only', 'none'] = 'full',
        ):
        """
        Split nodes based on batch information. Useful when you have multiple tissue sections
        and want to use entire batches/sections for different splits.

        Test batches are automatically assigned as all remaining batches after train and val.

        Parameters:
        ----------
        - region: dict or list of dicts
            Either a single region dict or a list of region dicts, where each region
            has keys 'x_min', 'x_max', 'y_min', 'y_max' defining regions to split on
        - xy_key: str
            The attribute name in the data object containing the xy coordinates
        - train_batches: list
            List of batch indices to assign to training. If None, all unassigned batches go to training.
        - val_batches: list
            List of batch indices to assign to validation.
        - test_batches: list
            List of batch indices to assign to testing.
        - train_val_cell_split: float
            Cell-level (not section-level) holdout fraction. When > 0,
            ~`train_val_cell_split` of cells in EACH training section are
            randomly moved from `train_mask` to `val_mask` so the val
            dataloader can compute an in-distribution early-stopping /
            best-checkpoint signal. The `val_*` wandb metrics then reflect
            the union of (a) any whole-section holdouts in `val_batches`
            and (b) this cell-level subsample of training sections.
            Useful when training is unstable late (e.g. adversarial
            losses) and you want best-ckpt-on-val to recover the best
            in-distribution snapshot. Default 0.0 = current behaviour
            (no cell-level split).
        - cell_split_seed: int
            RNG seed for the cell-level split. Same seed -> same split
            cells across runs (so wandb val_* curves are comparable
            across re-runs of the same variant).
        - test_regions: dict
            Per-batch test rectangles for the held-out-region
            reconstruction task. Maps `adata_batch_id -> region_spec`
            where each spec is either a single dict or a list of dicts.
            Supported region keys (mix freely; absolute and percentile
            entries can coexist on the same dict):
              absolute   : x_min, x_max, y_min, y_max  (in xy_coords)
              percentile : x_min_pct, x_max_pct, y_min_pct, y_max_pct
                           (fractions in [0,1] of the section's xy range)
            For batches present in this dict, the LISTED CELLS go to
            test_mask=True and the REST of THAT BATCH go to
            train_mask=True (val_mask=zero). Useful for the gene-
            reconstruction downstream task: the held-out patch is
            predicted by the trained model and Pearson is reported on
            test cells only. Independent from `test_batches` (which
            holds out whole sections); the two can be used together if
            needed.
        - holdout_graph: 'transductive' | 'drop' | 'rebuild'
            Graph used for the `test_regions` sections. 'transductive'
            (default, legacy) keeps the full graph, so held-out cells stay
            neighbours of observed cells. 'drop' and 'rebuild' rewrite
            `data.edge_index` with `make_inductive_edge_index` so that no
            edge goes from a held-out cell to an observed cell: 'drop'
            keeps the original observed -> observed edges, 'rebuild'
            replaces them with a kNN(`holdout_knn_k`) over observed cells
            only. Sections without a `test_regions` entry are untouched.
        - holdout_knn_k: int
            k for holdout_graph='rebuild' (required there; ignored
            otherwise). initialize_dataset_blob fills it from
            `graph_params['n_neighs']` when unset.
        - holdout_test_inedges: 'full' | 'test_only' | 'none'
            Edges INTO held-out cells under a non-transductive
            holdout_graph: 'full' (default) keeps all original in-edges,
            'test_only' only those from held-out cells, 'none' none.

        Example:
        --------
        # Single region
        region = {'x_min': 100, 'x_max': 150, 'y_min': 0, 'y_max': 100}

        # Multiple regions
        regions = [
            {'x_min': 0, 'x_max': 10, 'y_min': -10, 'y_max': 0},
            {'x_min': 15, 'x_max': 25, 'y_min': 5, 'y_max': 15}
        ]
        """
        self.region = region
        self.xy_key = xy_key
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.train_val_cell_split = float(train_val_cell_split)
        self.cell_split_seed = int(cell_split_seed)
        # Normalise keys to int so YAML-loaded `{15: {...}}` and
        # `{"15": {...}}` are both accepted.
        self.test_regions: Dict[int, Union[dict, List[dict]]] = (
            {int(k): v for k, v in (test_regions or {}).items()}
        )

        # Inductive held-out-region graph (opt-in). YAML nulls map to the
        # legacy defaults.
        holdout_graph = 'transductive' if holdout_graph is None else str(holdout_graph)
        holdout_test_inedges = 'full' if holdout_test_inedges is None else str(holdout_test_inedges)
        if holdout_graph not in HOLDOUT_GRAPH_MODES:
            raise ValueError(
                f"holdout_graph must be one of {HOLDOUT_GRAPH_MODES}, got {holdout_graph!r}"
            )
        if holdout_test_inedges not in HOLDOUT_TEST_INEDGES:
            raise ValueError(
                f"holdout_test_inedges must be one of {HOLDOUT_TEST_INEDGES}, "
                f"got {holdout_test_inedges!r}"
            )
        if holdout_graph == 'rebuild' and holdout_knn_k is None:
            raise ValueError("holdout_graph='rebuild' requires holdout_knn_k.")
        if holdout_knn_k is not None and int(holdout_knn_k) < 1:
            raise ValueError(f"holdout_knn_k must be positive, got {holdout_knn_k!r}")
        self.holdout_graph = holdout_graph
        self.holdout_knn_k = None if holdout_knn_k is None else int(holdout_knn_k)
        self.holdout_test_inedges = holdout_test_inedges
        if self.holdout_graph != 'transductive' and not self.test_regions:
            print(
                f"SpatialBatchSplit: holdout_graph={self.holdout_graph!r} has no "
                f"effect because test_regions is empty."
            )
        
    def _is_in_single_region(self, coords, region):
        """
        Check if coordinates are within a single specified region.

        Region can use ABSOLUTE bounds (`x_min`, `x_max`, `y_min`,
        `y_max`) and/or PERCENTILE bounds (`x_min_pct`, `x_max_pct`,
        `y_min_pct`, `y_max_pct` — fractions in [0, 1] of the
        section's actual xy range, computed on `coords`). Percentile
        keys make the same region spec portable across sections with
        different coordinate scales (MERFISH vs STARmap have very
        different absolute coordinate ranges).
        """
        if region is None:
            return torch.zeros(coords.shape[0], dtype=torch.bool)

        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        def _resolve_bound(abs_key: str, pct_key: str, fallback) -> float:
            if abs_key in region and region[abs_key] is not None:
                return float(region[abs_key])
            if pct_key in region and region[pct_key] is not None:
                rng_min = float(x_coords.min()) if abs_key.startswith("x") else float(y_coords.min())
                rng_max = float(x_coords.max()) if abs_key.startswith("x") else float(y_coords.max())
                pct = float(region[pct_key])
                return rng_min + pct * (rng_max - rng_min)
            return fallback

        x_min = _resolve_bound("x_min", "x_min_pct", float("-inf"))
        x_max = _resolve_bound("x_max", "x_max_pct", float("inf"))
        y_min = _resolve_bound("y_min", "y_min_pct", float("-inf"))
        y_max = _resolve_bound("y_max", "y_max_pct", float("inf"))

        x_mask = (x_coords >= x_min) & (x_coords <= x_max)
        y_mask = (y_coords >= y_min) & (y_coords <= y_max)

        return x_mask & y_mask
        
    def _is_in_region(self, coords, regions):
        """Check if coordinates are within any of the specified regions.
        
        Parameters:
        ----------
        coords : torch.Tensor
            Tensor of shape (N, 2) containing x,y coordinates
        regions : list or dict
            Either a single region dict or a list of region dicts, where each region
            has keys 'x_min', 'x_max', 'y_min', 'y_max'
        
        Returns:
        -------
        torch.Tensor
            Boolean tensor of shape (N,) indicating if each point is in any region
        """
        if regions is None:
            return torch.zeros(coords.shape[0], dtype=torch.bool)
            
        # Handle single region case
        if isinstance(regions, dict):
            return self._is_in_single_region(coords, regions)
            
        # Handle multiple regions case
        mask = torch.zeros(coords.shape[0], dtype=torch.bool)
        for region in regions:
            mask |= self._is_in_single_region(coords, region)
            
        return mask
    
    def _apply_inductive_graph(self, data: Data, test_mask: torch.Tensor, bid_int: int) -> None:
        """
        Replace `data.edge_index` in place with the inductive graph from
        `make_inductive_edge_index` and print what changed.
        """
        old_ei = data.edge_index
        num_nodes = int(test_mask.numel())
        t = test_mask.to(device=old_ei.device, dtype=torch.bool)
        new_ei = make_inductive_edge_index(
            edge_index=old_ei,
            test_mask=test_mask,
            xy=data[self.xy_key],
            mode=self.holdout_graph,
            k=self.holdout_knn_k,
            test_inedges=self.holdout_test_inedges,
        )

        # Bookkeeping for the log line: edges removed / added (as sets),
        # the observed boundary band (observed cells with >= 1 held-out
        # in-neighbour in the ORIGINAL graph) and the held-out ring.
        old_l = old_ei.to(torch.long)
        new_l = new_ei.to(torch.long)
        old_key = old_l[0] * num_nodes + old_l[1]
        new_key = new_l[0] * num_nodes + new_l[1]
        n_removed = int((~torch.isin(old_key, new_key)).sum())
        n_added = int((~torch.isin(new_key, old_key)).sum())
        cross = t[old_l[0]] & ~t[old_l[1]]
        n_band = int(torch.unique(old_l[1][cross]).numel())
        n_ring = int(torch.unique(old_l[0][cross]).numel())

        data.edge_index = new_ei
        # SetExperimentDataKeys stores num_edges explicitly; keep it in sync.
        if 'num_edges' in data.keys():
            data.num_edges = int(new_ei.size(1))

        k_str = f", k={self.holdout_knn_k}" if self.holdout_graph == 'rebuild' else ""
        print(
            f"SpatialBatchSplit: batch {bid_int} holdout_graph="
            f"{self.holdout_graph!r}{k_str}, test_inedges="
            f"{self.holdout_test_inedges!r}: edges {int(old_ei.size(1))} -> "
            f"{int(new_ei.size(1))} (removed {n_removed}, added {n_added}); "
            f"n_test={int(t.sum())}, observed boundary band={n_band} cells, "
            f"held-out ring={n_ring} cells."
        )

    def forward(self, data: Data) -> Data:
        num_nodes = data.x.shape[0]

        # Per-batch held-out test regions (independent of `test_batches`).
        # When this batch has an entry in `test_regions`, cells inside the
        # region(s) are test_mask=True, the REMAINING cells go through
        # the same in-section cell-level train/val split as a normal
        # training section (so early-stopping on val_loss still has a
        # signal — the val cells are NOT held-out-region cells, they're
        # a random 10% sample of the rest of the section).
        # Used by the held-out gene-reconstruction downstream task.
        bid_int = int(data.adata_batch_id)
        if self.test_regions and bid_int in self.test_regions:
            region_spec = self.test_regions[bid_int]
            test_mask = self._is_in_region(
                data[self.xy_key], region_spec,
            )
            train_mask = ~test_mask
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)

            # Cell-level train/val split on the non-test cells. Same RNG
            # convention as the train_batches branch
            # (`(cell_split_seed, adata_batch_id)` -> deterministic).
            if self.train_val_cell_split > 0.0:
                gen = torch.Generator()
                gen.manual_seed(
                    self.cell_split_seed * 1_000_003 + int(data.adata_batch_id)
                )
                train_idx = torch.where(train_mask)[0]
                n_eligible = int(train_idx.numel())
                if n_eligible > 0:
                    n_val = int(round(self.train_val_cell_split * n_eligible))
                    perm = torch.randperm(n_eligible, generator=gen)
                    val_positions = train_idx[perm[:n_val]]
                    val_mask[val_positions] = True
                    train_mask[val_positions] = False

            data.test_mask  = test_mask
            data.train_mask = train_mask
            data.val_mask   = val_mask

            # Opt-in inductive graph: cut every held-out -> observed edge
            # so held-out counts never reach observed cells (train, val
            # and predict all re-apply this transform). Masks and node
            # order are unchanged.
            if self.holdout_graph != 'transductive':
                self._apply_inductive_graph(data, test_mask, bid_int)
            return data

        if data.adata_batch_id in self.train_batches:
            data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            # Cell-level train/val split. Random `train_val_cell_split`
            # fraction (default 0.10) of cells in each training section
            # is moved from train_mask -> val_mask, providing the val
            # signal used during training for early stopping +
            # best-checkpoint selection. RNG is seeded with
            # `(cell_split_seed, adata_batch_id)` so the partition is
            # deterministic across runs and uncorrelated across sections
            # (different sections get different sub-samples even though
            # the seed is the same).
            #
            # Note: at inference time, val cells are folded back into
            # "train" — see `data_split` in
            # `_build_clean_adata_from_inference`. The val set is purely
            # a training-time mechanism.
            if self.train_val_cell_split > 0.0:
                gen = torch.Generator()
                gen.manual_seed(
                    self.cell_split_seed * 1_000_003
                    + int(data.adata_batch_id)
                )
                perm = torch.randperm(num_nodes, generator=gen)
                n_val = int(round(self.train_val_cell_split * num_nodes))
                val_idx = perm[:n_val]
                data.val_mask[val_idx]   = True
                data.train_mask[val_idx] = False

        elif data.adata_batch_id in self.val_batches:
            # `region=None` means "this entire batch is held out as val"
            # (used by holdout-replicate variants where whole AnnData files
            # are kept aside for evaluation). When a region IS given, only
            # cells inside the region are val, the rest train.
            if self.region is None:
                data.val_mask = torch.ones(num_nodes, dtype=torch.bool)
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                data.val_mask = self._is_in_region(data.xy_coordinates, self.region)
                data.train_mask = ~data.val_mask
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        elif data.adata_batch_id in self.test_batches:
            # Same convention as val_batches above.
            if self.region is None:
                data.test_mask = torch.ones(num_nodes, dtype=torch.bool)
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                data.test_mask = self._is_in_region(data.xy_coordinates, self.region)
                data.train_mask = ~data.test_mask
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)

        return data


class SubsetHVG(T.BaseTransform):
    def __init__(
            self,
            n_genes: int = 1000,
        ):
        """
        Subset the raw gene counts matrix to the top n_genes highly variable genes.

        Parameters:
        ----------
        - n_genes: int
            The number of highly variable genes to subset the data to.
        """
        self.n_genes = n_genes

    def forward(
            self,
            data: Data
        ) -> Data:
        """
        This transform is applied to the raw gene counts matrix.

        We also stash `data.hvg_indices` (LongTensor of length n_genes) — the
        positions in the original `data.x_cell_gene_counts` columns that
        survived HVG selection. Downstream code (e.g. predict()) can use this
        together with the dataset blob's saved gene_panel.pkl to recover gene
        names for the post-HVG feature matrix.
        """
        adata = ad.AnnData(data.x_cell_gene_counts.numpy())
        sc.pp.highly_variable_genes(
            adata,
            flavor='seurat_v3',
            n_top_genes=self.n_genes,
            subset=True
        )
        # we manually set the data.x to the subsetted adata.X
        data.x = torch.from_numpy(adata.X)
        # adata.var.index here is the integer positions (as strings) into the
        # original x_cell_gene_counts columns, since we built the temp adata
        # from a name-less ndarray.
        data.hvg_indices = torch.tensor(
            [int(v) for v in adata.var.index], dtype=torch.long
        )
        print(f"SubsetHVG: Subsetted data to {data.num_features} features.")
        return data


class NormalizeFeatures(T.BaseTransform):
    def __init__(
            self,
            norm_method: str = 'read_depth',
            target_size: int = 10_000,
            apply_CPM: Optional[bool] = True
        ):
        """
        Normalize the features of the PyG Data object.

        Parameters:
        ----------
        - norm_method: str
            The method to use for normalization.
        - target_size: int
            The target size for normalization.
        - apply_CPM: bool
            If True, apply CPM normalization.

        Notes:
        -----
        Ideally, the feature_key should be 'x' for the node features. The option exists to pass a custom key to allow for flexibility during training, if required.
        """
        self.norm_method = norm_method
        self.target_size = target_size
        self.apply_CPM = apply_CPM

    def forward(
            self,
            data: Data
        ) -> Data:
        """
        Normalize the features of the PyG Data object.
        """
        # if SubsetHVG is applied, the feature data to normalize is data.x
        if 'x' in data.keys():
            feature_data = data.x
        # if SubsetHVG is not applied, the feature data to normalize is the raw cell gene counts in data.x_cell_gene_counts
        else:
            feature_data = data.x_cell_gene_counts

        if self.norm_method == 'read_depth':
            feature_data = normalize_by_read_depth(
                            x=feature_data,
                            target_size=self.target_size
                        )
        elif self.norm_method == 'total_log1p':
            feature_data = normalize_total_log1p(
                            x=feature_data,
                            target_size=self.target_size,
                            apply_CPM=self.apply_CPM
                        )
        else:
            raise ValueError(f"Normalization method {self.norm_method} not found.")

        # the normalized feature data is stored in data.x
        setattr(data, 'x', feature_data)

        return data


class SetExperimentDataKeys(T.BaseTransform):
    def __init__(
            self,
            feature_names: List[Literal['X', 'U_lm_eigvecs', 'U_deepwalk', 'U_gosh']] = ['X'],
            label_name: str = 'cell_types',
            edge_index_name: str = 'spatial-delaunay',
            encoder_condition_list: Optional[List[str]] = None,
            spatial_prior_feature: Optional[str] = None,
            attr_decoder_condition_list: Optional[List[str]] = None,
            adj_decoder_condition_list: Optional[List[str]] = None,
        ):
        """
        Set data.x, data.y, and data.edge_index keys for the PyG Data object from Experiment keys.
        This is useful to control before starting training so we can choose on the fly which features, labels, and edge indices to use (e.g., cell gene counts for cell type classification, etc.)

        Parameters:
        ----------
        - feature_names: List[Literal['X', 'U_lm_eigvecs', 'U_deepwalk', 'U_gosh']]
            The keys for the node features to set.
        - label_name: Literal['cell_types']
            The key for the node labels to set.
        - edge_index_name: Literal['spatial-delaunay']
            The key for the edge index to set.
        - encoder_condition_list: List[str]
            List of condition names to be used for conditioning the encoder.
        - spatial_prior_feature: str
            Feature to be used for training the spatial prior for the codebook.
        - attr_decoder_condition_list: List[str]
            List of condition names to be used for conditioning the attribute decoder.
        - adj_decoder_condition_list: List[str]
            List of condition names to be used for conditioning the adjacency decoder.

        Notes:
        -----
        - X refers to the raw cell gene counts.
        - U_lm_eigvecs, U_deepwalk, U_gosh refer to the unsupervised node embeddings generated by the corresponding methods.
        - If feature_names contains multiple names, the features are concatenated along the feature dimension.
        """
        self.feature_names = feature_names
        self.label_name = label_name
        self.edge_index_name = edge_index_name
        self.encoder_condition_list = encoder_condition_list
        self.spatial_prior_feature = spatial_prior_feature
        self.attr_decoder_condition_list = attr_decoder_condition_list
        self.adj_decoder_condition_list = adj_decoder_condition_list


    def set_node_attributes(
            self,
            data: Data,
        ) -> torch.Tensor:
        """
        Set the node attributes in the data object given the feature_name.

        Parameters:
        ----------
        - data: Data
            The data object to set the node attributes in.

        Returns:
        -------
        - torch.Tensor
            The node attributes.
        """
        feature_keys = []
        for feature_name in self.feature_names:
            if feature_name == 'X':
                # if SubsetHVG and/or NormalizeFeatures is applied, the feature data is in data.x
                if 'x' in data.keys():
                    feature_keys.append('x')
                # if SubsetHVG and/or NormalizeFeatures is not applied, the feature data is in data.x_cell_gene_counts
                else:
                    feature_keys.append('x_cell_gene_counts')
            elif feature_name in ['U_lm_eigvecs', 'U_deepwalk', 'U_gosh']:
                feature_keys.append(f"{feature_name}_{self.edge_index_name}")
            else:
                raise ValueError(f"Feature key {feature_name} not found in data.")

        x = torch.cat(
                [getattr(data, key) for key in feature_keys],
                dim=1
            )
        return x


    def set_node_labels(
            self,
            data: Data,
        ) -> torch.Tensor:
        """
        Set the node labels in the data object given the label_name.

        When `label_name` is None (no supervision configured) or when the
        named label is absent from this batch (e.g. the dataset blob was
        built with `label_names=[]` because the source AnnDatas don't
        carry a usable label column), return a zero-channel placeholder
        tensor of shape `(N, 0)`. Downstream code uses `data.y.shape[1]`
        as `num_classes`; a 0-channel tensor gives `num_classes=0`, which
        is a valid no-supervision signal and lets the model be built and
        trained on purely-unsupervised objectives (NB recon, adjacency,
        commit, adversarial). The cross-entropy loss is never registered
        in loss_names for these configs, so the dummy `data.y` is unused.
        """
        if self.label_name is None or f"y_{self.label_name}" not in data.keys():
            num_nodes = (
                data.x.shape[0] if hasattr(data, 'x')
                else data.num_nodes
            )
            return torch.zeros(num_nodes, 0)
        return getattr(data, f"y_{self.label_name}")


    def set_edge_index(
            self,
            data: Data,
        ) -> torch.Tensor:
        """
        Set the edge index in the data object given the edge_index_name.
        """
        if f"edge_index_{self.edge_index_name}" in data.keys():
            return getattr(data, f"edge_index_{self.edge_index_name}")
        else:
            raise ValueError(f"Edge index key {self.edge_index_name} not found in data.")


    def set_conditioning_features(
            self,
            data: Data,
            condition_list: List[str] = [],
        ) -> torch.Tensor:
        """
        Set the conditioning features in the data object given the a list of condition names.
        """
        conditioning_features = []

        for source in condition_list:
            if source == 'absolute_xy':
                conditioning_features.append(data.xy_coordinates)

            elif source == 'fourier_xy':
                conditioning_features.append(
                    fourier_encode(
                        data.xy_coordinates,
                    )
                )

            elif source == 'relative_xy':
                centroid = data.xy_coordinates.mean(dim=0, keepdim=True)
                rel_coords = data.xy_coordinates - centroid
                conditioning_features.append(rel_coords)

            elif source == 'rbf_distances':
                # Compute distance from centroid
                centroid = data.xy_coordinates.mean(dim=0, keepdim=True)
                dists = torch.norm(data.xy_coordinates - centroid, dim=1)  # (N,)
                centers = torch.linspace(dists.min(), dists.max(), steps=8).to(dists.device)
                rbf_feats = rbf_encode(dists, centers, gamma=10.0)
                conditioning_features.append(rbf_feats)
                
            elif source == 'cell_types':
                conditioning_features.append(data.y)

            elif source == 'U_lm_eigvecs':
                conditioning_features.append(getattr(data, f"U_lm_eigvecs_{self.edge_index_name}"))

            elif source == 'U_deepwalk':
                conditioning_features.append(getattr(data, f"U_deepwalk_{self.edge_index_name}"))

            elif source == 'U_gosh':
                conditioning_features.append(getattr(data, f"U_gosh_{self.edge_index_name}"))
                
            elif source == 'cell_batch_id':
                conditioning_features.append(torch.empty(0))
                
            elif source == 'timepoint_id':
                conditioning_features.append(torch.empty(0))
                
            elif source == 'grade':
                conditioning_features.append(data.y_grade)

        conditioning_features = torch.cat(conditioning_features, dim=-1)

        return conditioning_features


    def set_spatial_prior_features(
            self,
            data: Data,
            feature_name: str = None,
        ) -> torch.Tensor:
        """
        Set the spatial prior features in the data object given the feature name.
        """
        if feature_name == 'fourier_xy':
            return fourier_encode(
                    data.xy_coordinates,
                )
            
        elif feature_name == 'rbf_distances':
            # Compute distance from centroid
            centroid = data.xy_coordinates.mean(dim=0, keepdim=True)
            dists = torch.norm(data.xy_coordinates - centroid, dim=1)  # (N,)
            centers = torch.linspace(dists.min(), dists.max(), steps=8).to(dists.device)
            rbf_feats = rbf_encode(dists, centers, gamma=10.0)
            return rbf_feats

        else:
            raise ValueError(f"Spatial prior feature {feature_name} not found in data.")


    def forward(
            self,
            data: Data
        ) -> Data:
        """
        Set Experiment data keys for the PyG Data object.
        """
        data.x = self.set_node_attributes(data)
        data.y = self.set_node_labels(data)
        data.edge_index = self.set_edge_index(data)

        if self.encoder_condition_list is not None:
            print(f"Setting section-level conditioning features for encoder.")
            data.encoder_conditions = self.set_conditioning_features(
                data=data,
                condition_list=self.encoder_condition_list,
            )
            if data.encoder_conditions.dim() > 1:
                data.encoder_condition_dim = data.encoder_conditions.shape[1]
            else:
                data.encoder_condition_dim = 0
        else:
            data.encoder_conditions = None
            data.encoder_condition_dim = 0
            
        if self.spatial_prior_feature is not None:
            print(f"Setting spatial prior features for encoder.")
            data.spatial_prior_features = self.set_spatial_prior_features(
                data=data,
                feature_name=self.spatial_prior_feature,
            )
            data.spatial_prior_feature_dim = data.spatial_prior_features.shape[1]
        else:
            data.spatial_prior_features = None
            data.spatial_prior_feature_dim = 0
            
        if self.attr_decoder_condition_list is not None:
            print(f"Setting section-level conditioning features for attribute decoder.")
            data.attr_decoder_conditions = self.set_conditioning_features(
                data=data,
                condition_list=self.attr_decoder_condition_list,
            )
            if data.attr_decoder_conditions.dim() > 1:
                data.attr_decoder_condition_dim = data.attr_decoder_conditions.shape[1]
            else:
                data.attr_decoder_condition_dim = 0
        else:
            data.attr_decoder_conditions = None
            data.attr_decoder_condition_dim = 0
        
        if self.adj_decoder_condition_list is not None:
            print(f"Setting section-level conditioning features for adjacency decoder.")
            data.adj_decoder_conditions = self.set_conditioning_features(
                data=data,
                condition_list=self.adj_decoder_condition_list,
            )
            if data.adj_decoder_conditions.dim() > 1:
                data.adj_decoder_condition_dim = data.adj_decoder_conditions.shape[1]
            else:
                data.adj_decoder_condition_dim = 0
        else:
            data.adj_decoder_conditions = None
            data.adj_decoder_condition_dim = 0

        data.num_features = data.x.shape[1]
        data.num_classes = data.y.shape[1]
        data.num_nodes = data.x.shape[0]
        data.num_edges = data.edge_index.shape[1]

        # delete extra features, edge indices, and embeddings from the data object to reduce memory footprint during training
        for key in list(data.keys()):
            if key.startswith('x_') or key.startswith('edge_index_') or key.startswith('U_'):
                delattr(data, key)

        return data
    
    
def fourier_encode(coords, num_freqs=6):
    """
    Sinusoidal positional encoding of 2D coordinates.
    coords: Tensor of shape (N, 2)
    returns: (N, 4 * num_freqs)
    """
    freq_bands = 2 ** torch.arange(num_freqs, device=coords.device) * torch.pi
    coords = coords.unsqueeze(-1)  # (N, 2, 1)
    sin = torch.sin(freq_bands * coords)  # (N, 2, F)
    cos = torch.cos(freq_bands * coords)  # (N, 2, F)
    enc = torch.cat([sin, cos], dim=-1)   # (N, 2, 2F)
    return enc.view(coords.size(0), -1)   # (N, 4F)

def rbf_encode(distances, centers, gamma):
    """
    RBF encoding: Gaussian basis functions centered at given values.
    distances: (N, )
    centers: (R, )
    returns: (N, R)
    """
    diff = distances.unsqueeze(1) - centers.view(1, -1)
    return torch.exp(-gamma * (diff ** 2))