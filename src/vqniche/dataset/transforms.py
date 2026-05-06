from typing import Optional, List, Literal, Union

import scanpy as sc
import anndata as ad

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from ..preprocessors.normalizers import normalize_by_read_depth, normalize_total_log1p


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
        test_batches: Optional[list] = None
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
                test_batches=test_batches
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
            test_batches: Optional[list] = None
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
        
    def _is_in_single_region(self, coords, region):
        """Check if coordinates are within a single specified region."""
        if region is None:
            return torch.zeros(coords.shape[0], dtype=torch.bool)
            
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        x_mask = (x_coords >= region['x_min']) & (x_coords <= region['x_max'])
        y_mask = (y_coords >= region['y_min']) & (y_coords <= region['y_max'])
        
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
    
    def forward(self, data: Data) -> Data:
        num_nodes = data.x.shape[0]

        if data.adata_batch_id in self.train_batches:
            data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

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