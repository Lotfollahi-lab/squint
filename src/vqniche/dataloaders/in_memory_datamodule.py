"""
This class extends the Pytorch Lightning's LightningDataModule class to provide a more flexible way to create PyTorch Lightning DataModules that are compatible with PyTorch Geometric for our use case wherein we split the Pytorch Geometric GNN models into separate encoder and predictor submodules.
This has two main advantages:
1. We can inherit all the nice properties of the parent classes including `setup`, `prepare_data` and `infer train-val-test` nodes. These can also be overriden in the future if necessary.
2. We can define our custom train, validation, test, and inference dataloaders that are compatible with PyTorch Geometric's DataLoader and Sampler classes.

Initializing this class requires defining a Loader and a Sampler along with their corresponding parameters.

The `loader_name` argument is used to define the DataLoader class that will be used to load the data. Currently, we support the following:
- `FullLoader`: This option provides backwards compatibility with the default `LightningNodeData` class that uses the full graph for training.
- `DefaultNodeLoader`: This option provides backwards compatibility with the default `LightningNodeData` class that uses the default `NodeLoader` along with `NeighborSampler` from PyTorch Geometric.
- NeighborLoader: This option uses the `NeighborLoader` class from PyTorch Geometric. Ignores the loader and sampler related setup in the parent class.
The `loader_params` argument is a dictionary that can be used to define the arguments that will be passed to the DataLoader class. We mainly focus on `batch_size` for now.
Use `FullLoader` and `DefaultNodeLoader` only if the default `LightningNodeData` class is sufficient for your use case. Otherwise, define and use your own custom Loader and Sampler.

The `sampler_name` argument is used to define the Sampler class that will be used to sample the subgraphs. Currently, we support the following:
- `NeighborSampler`: This option uses the `NeighborSampler` class from PyTorch Geometric.
The `sampler_params` argument is a dictionary that can be used to define the arguments that will be passed to the Sampler class. We mainly focus on `num_neighbors` for now.

KEY INFO:
---> The Loader + Sampler combination is kept the same across training, validation and testing.
---> The input data to this class must be a single PyTorch Geometric Data object. If the experiment requires multiple tissue sections, the data must be concatenated before being passed to this class.
---> The `infer_dataloader` method is provided to allow for looping over all the nodes in the graph unlike the training, validation, and testing dataloaders which only loop over the nodes in the training, validation, and test sets respectively.
---> The `sample_neighbors_for_inference` parameter is used to control whether the neighbors are sampled during inference or not. This parameter is the same for validation and testing and is ignored for training.
"""
import os
import time
from typing import Literal, Optional, Callable, List, Any

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NodeLoader, NeighborLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.data.lightning import LightningNodeData


def _identity_collate(items: List[Any]) -> Any:
    """
    Collate fn for the cached-batch DataLoader.

    Each item in the underlying list IS already a fully-materialised
    `torch_geometric.data.Batch` (built once during pre-caching). The
    DataLoader is configured with `batch_size=1` so `items` is always a
    one-element list — we just unwrap it. Using a real function (not a
    lambda) keeps this picklable in case Lightning ever wraps the
    loader for spawn-based DDP.
    """
    return items[0]

NUM_CORES = 1
NUM_WORKERS = 1
BATCH_SIZE = 1024
NUM_NEIGHBORS = [5, 5]


class InMemoryDataModule(LightningNodeData):
    def __init__(
            self,
            data: Data,
            loader_name: Literal['DefaultFullLoader', 'DefaultNodeLoader', 'NeighborLoader'] = 'NeighborLoader',
            loader_params: Optional[dict] = {},
            sampler_name: Optional[Literal['NeighborSampler']] = 'NeighborSampler',
            sampler_params: Optional[dict] = {},
            sample_neighbors_for_inference: bool = False,
            predict_sample_neighbors_for_inference: bool = False,
            predict_batch_size: Optional[int] = None,
            val_batch_size: Optional[int] = None,
            cache_train_batches: bool = False,
            obs_per_batch_id: Optional[dict] = None,
        ) -> None:
        """
        This function initializes the InMemoryDataModule class with the given DataLoader and Sampler classes along with their corresponding parameters.

        Parameters:
        -----------
        - data: Data
            A single PyTorch Geometric Data object that contains the graph data.
        - loader_name: Literal['DefaultFullLoader', 'DefaultNodeLoader', 'NeighborLoader']
            Name of the DataLoader class that will be used to load the data.
        - loader_params: Optional[dict]
            Dictionary that contains the arguments that will be passed to the DataLoader class, e.g. `batch_size`.
        - sampler_name: Optional[Literal['NeighborSampler']]
            Name of the Sampler class that will be used to sample the subgraphs.
        - sampler_params: Optional[dict]
            Dictionary that contains the arguments that will be passed to the Sampler class, e.g. `num_neighbors`.
        - sample_neighbors_for_inference: bool
            Whether to sample neighbors for the VALIDATION and TEST
            dataloaders. When False (legacy default), val/test fan out
            with `num_neighbors=[-1]` (every neighbor). When True, val/test
            re-use the training sampler's `num_neighbors` so the GPU
            stays fed during validation (the per-step cost matches
            training instead of being 10-100x heavier).
        - predict_sample_neighbors_for_inference: bool
            Independent toggle for the PREDICT dataloader. Default
            `False`: the final predict pass always fans out with
            `num_neighbors=[-1]` so the embeddings written to AnnData
            are deterministic and aggregate every neighbor — that's
            what downstream batch-integration metrics (iLISI, MMD)
            and clustering metrics (NMI, ARI) expect. Set to `True`
            only if you specifically want stochastic predict-time
            sampling (e.g. dropout-style uncertainty estimates).

            The split exists because val/test can legitimately want
            cheap sampling for training-loop throughput, while the
            single final predict pass should be heavy + deterministic.
            Setting `sample_neighbors_for_inference=True` alone (the
            common case) leaves predict on the safe default.
        - predict_batch_size: Optional[int]
            Per-batch size used by the PREDICT dataloader only. When
            None (legacy default), predict inherits the training
            `batch_size` from `loader_params`. The training batch size
            is tuned for adversarial gradient dynamics — predict has
            no such constraint and benefits from a much bigger batch
            (fewer kernel launches, better GPU SM utilisation).
            Combined with `predict_sample_neighbors_for_inference=False`
            (the full-fanout default), bumping this to 4-8x the
            training batch size typically cuts predict wall-clock 2-3x.
            Memory ceiling: bigger predict batches carry more sampled
            neighbours per batch ([-1] fanout × bigger seeds = bigger
            total node count). 1024 is known-safe on a 256 GiB GPU
            for the mmb-smb graph; bump to 2048+ on larger memory.
        - val_batch_size: Optional[int]
            Per-batch size used by the VAL and TEST dataloaders. When
            None (legacy default), val/test inherit the training
            `batch_size` from `loader_params`. Like predict, val and
            test are no-grad passes so the small training batch is
            unnecessary — bumping to 4-8x the training batch reduces
            the val-pass wall-clock at every val epoch, which directly
            lifts the GPU-utilisation floor (the GPU previously sat
            idle for several seconds per val pass as small val batches
            dribbled through). Combined with `check_val_every_n_epoch=2`
            in the trainer config, this is one of the cheapest
            throughput wins for SQUINT training. Safe to set
            independently from `predict_batch_size`; val typically
            uses the val-loader's own fanout (`sample_neighbors_for_inference`),
            which is lighter than predict's `[-1]` so val can usually
            go even bigger than predict.
        - cache_train_batches: bool
            When True, the TRAINING DataLoader pre-computes one full
            epoch's worth of `Batch` objects on first access and
            serves the same cached list every subsequent epoch. The
            standard NeighborLoader path (CPU NeighborSampler + induced
            subgraph construction + tensor copy) is the dominant CPU
            cost in SQUINT training; caching collapses it to a single
            up-front pass and turns every later epoch into a pure
            memcpy + H2D-copy loop. Per-epoch reshuffling of BATCH
            ORDER is preserved (the cached list is wrapped in a
            DataLoader with `shuffle=True`), but per-epoch neighbor
            resampling and per-epoch seed-to-batch regrouping are
            NOT — every epoch sees the same set of batches.

            IMPORTANT: when caching is on, the cache-build loader's
            `num_neighbors` is overridden to `[-1]` (full neighborhood)
            regardless of `sampler_params['num_neighbors']`. Rationale:
            the cache freezes neighborhoods anyway, so freezing the
            FULL per-cell neighborhood is strictly more informative
            than freezing a random subset — same staticness either
            way. `self.sampler_params` is NOT mutated; val/test/predict
            loaders still read the original fanout.

            Trade-off:
              + 30-60% wall-clock saving on CPU-sampler-heavy datasets
                (SQUINT-scale spatial graphs with fanout=[8]).
              + Lower CPU pressure → frees workers for other jobs on
                shared nodes.
              + Each cell sees its FULL spatial neighborhood every
                step (richer signal than the sampled-[8] fanout used
                by the uncached path).
              - Loss of per-epoch neighbor-resampling regularisation.
                For SQUINT this is usually small because the spatial
                k-NN graph is sparse (avg degree ~10-12) — the
                augmentation lost is small. A/B test on niche NMI /
                iLISI if you're worried.
              - Memory: stores ~N_train_batches × per-batch-node-tensors
                in CPU memory. With the full-fanout override, ~2-3 GB
                at SQUINT-mmb-smb scale (175 batches × ~12 neighbors ×
                256 seeds × 431 genes float32), more like ~5-15 GB on
                chl59-scale graphs. Still comfortable inside the
                128 GB LSF default for the mmb / smb panels; consider
                turning OFF on chl59 if you tighten memory limits.

            Defaults to False to preserve legacy behaviour; turn on
            via the base config or per-variant patches.
        """
        assert isinstance(data, Data), f"data must be of type torch_geometric.data.Data, but got {type(data)}."

        # get the number of available cores
        num_cores_available = int(os.environ.get(
                                "LSB_DJOB_NUMPROC",
                                NUM_CORES
                                ))

        # Honour an explicit `num_workers` passed via `loader_params` if
        # present. Pop it here BEFORE the per-loader branch logic so it
        # (a) survives the per-branch `loader_params` reassignment below
        # (DefaultFullLoader / DefaultNodeLoader both rebuild
        # `loader_params` from scratch, which would otherwise drop the
        # caller's value), and (b) isn't passed twice to
        # `super().__init__()` — the parent class declares `num_workers`
        # as an explicit kwarg AND we expand `**self.loader_params`
        # there, so leaving `num_workers` in both would raise
        # `TypeError: got multiple values for keyword argument 'num_workers'`.
        # Note: `DefaultFullLoader` deliberately forces `num_workers=0`
        # (incompatible with full-graph loading) — the explicit override
        # is NOT honoured for that path.
        explicit_num_workers = (
            loader_params.pop('num_workers', None)
            if isinstance(loader_params, dict)
            else None
        )

        # Loaders will be instantiated in self.train_dataloader(), self.val_dataloader(), and self.test_dataloader()
        print(f"Loader Name: {loader_name}")

        # setting parameters for backward compatibility with the default LightningNodeData implementation of full graph training
        if loader_name == 'DefaultFullLoader':
            # set num_workers to 0 for FullLoader
            num_workers = 0

            # set loader_type to 'full' for FullLoader
            loader_type = 'full'
            loader_class: Callable = DataLoader

            # batch_size = 1 is the only parameter for FullLoader
            loader_params = {'batch_size': 1}

            # sampler must be set to None for FullLoader
            sampler_name = None
            sampler_class: Callable = None

            # sampler_params must be set to an empty dictionary for FullLoader
            sampler_params = {}

        # setting parameters for backward compatibility with the default LightningNodeData implementation of NodeLoader + NeighborSampler
        elif loader_name == 'DefaultNodeLoader':
            # set num_workers to half of the available cores (or honour
            # the caller-pinned `loader_params['num_workers']` popped
            # above, if any).
            num_workers = (
                explicit_num_workers
                if explicit_num_workers is not None
                else max(num_cores_available // 2, NUM_WORKERS)
            )

            # set loader_type to 'neighbor' for DefaultNodeLoader
            loader_type = 'neighbor'
            loader_class: Callable = NodeLoader

            # set batch_size to BATCH_SIZE if not provided in loader_params and reset loader_params to only include batch_size
            loader_params['batch_size'] = loader_params.get(
                                            'batch_size',
                                            BATCH_SIZE
                                            )
            loader_params = {'batch_size': loader_params['batch_size']}

            # LightningNodeData will reset node_sampler to torch_geometric.loader.NeighborSampler and initialize it with the given sampler kwargs.
            # this will be gettable via the self.graph_sampler attribute.
            sampler_name = None
            sampler_class: Callable = None

            # set num_neighbors to NUM_NEIGHBORS if not provided in sampler_params and reset sampler_params to only include num_neighbors
            sampler_params['num_neighbors'] = sampler_params.get(
                                                'num_neighbors',
                                                NUM_NEIGHBORS
                                                )
            sampler_params = {'num_neighbors': sampler_params['num_neighbors']}

        # setting parameters for custom Loader and Sampler
        else:
            # set num_workers to half of the available cores (or honour
            # the caller-pinned `loader_params['num_workers']` popped
            # above, if any).
            num_workers = (
                explicit_num_workers
                if explicit_num_workers is not None
                else max(num_cores_available // 2, NUM_WORKERS)
            )

            # train_loader will be set to a custom Callable in self.train_dataloader()
            # this is necessary to indicate to LightningNodeData that the train_loader is a custom DataLoader
            loader_type = 'custom'
            loader_class: Callable = self.set_custom_loader_class(loader_name)

            # set batch_size to BATCH_SIZE if not provided in loader_params
            loader_params['batch_size'] = loader_params.get(
                                            'batch_size',
                                            BATCH_SIZE
                                            )

            # set node_sampler to be a Callable of type torch_geometric.sampler.BaseSampler here before super.__init__() is called.
            # cannot be None because LightningNodeData will throw an error when loader_type is `custom`.
            # this will be gettable via the self.graph_sampler attribute (but we don't use it)
            sampler_class: Callable = self.set_custom_sampler_class(sampler_name)

            # set num_neighbors to NUM_NEIGHBORS if not provided in sampler_params
            sampler_params['num_neighbors'] = sampler_params.get(
                                                'num_neighbors',
                                                NUM_NEIGHBORS
                                                )

        # set the backend class attributes
        self.num_workers = num_workers
        print(f"Num Workers: {num_workers}")

        # set the loader attributes
        self.loader_name = loader_name
        self.loader_class = loader_class
        self.loader_params = loader_params
        print(f"Loader Name: {self.loader_name}")
        print(f"Loader Class: {self.loader_class}")
        print(f"Loader Params: {self.loader_params}")

        # set the sampler attributes
        self.sampler_name = sampler_name
        self.sampler_class = sampler_class
        self.sampler_params = sampler_params
        print(f"Sampler Name: {self.sampler_name}")
        print(f"Sampler Class: {self.sampler_class}")
        print(f"Sampler Params: {self.sampler_params}")

        # set whether to sample neighbors for inference. val/test and
        # the final predict pass each get their own switch — see the
        # constructor docstring for the rationale.
        self.sample_neighbors_for_inference = sample_neighbors_for_inference
        self.predict_sample_neighbors_for_inference = predict_sample_neighbors_for_inference
        # predict-only batch-size override (see constructor docstring).
        # `None` means "inherit `loader_params['batch_size']`" — the
        # legacy behaviour. Validated here so a string config typo
        # ("1024" vs 1024) fails loudly rather than poisoning predict.
        if predict_batch_size is not None and int(predict_batch_size) <= 0:
            raise ValueError(
                f"predict_batch_size must be positive when set; "
                f"got {predict_batch_size!r}"
            )
        self.predict_batch_size = (
            int(predict_batch_size) if predict_batch_size is not None else None
        )
        # val/test batch-size override (same legacy-safe shape as
        # predict_batch_size). Sized independently from predict because
        # val/test use a different fanout (`sample_neighbors_for_inference`,
        # typically `[8]`) vs. predict (`[-1]`), and the val pass runs
        # every N epochs, so per-pass cost is the bottleneck not memory.
        if val_batch_size is not None and int(val_batch_size) <= 0:
            raise ValueError(
                f"val_batch_size must be positive when set; "
                f"got {val_batch_size!r}"
            )
        self.val_batch_size = (
            int(val_batch_size) if val_batch_size is not None else None
        )
        # Pre-cache one epoch of TRAIN batches on first `train_dataloader()`
        # access. See constructor docstring for the full trade-off; default
        # is False so legacy callers behave as before.
        self.cache_train_batches = bool(cache_train_batches)
        # Lazy slot — populated on the first `train_dataloader()` call when
        # caching is enabled. None means "not yet built".
        self._cached_train_batches: Optional[list] = None
        print(f"Sample Neighbors for Inference (val/test): "
              f"{self.sample_neighbors_for_inference}")
        print(f"Sample Neighbors for Inference (predict):  "
              f"{self.predict_sample_neighbors_for_inference}")
        print(f"Predict Batch Size:                        "
              f"{self.predict_batch_size if self.predict_batch_size is not None else '(inherit from loader_params)'}")
        print(f"Val/Test Batch Size:                       "
              f"{self.val_batch_size if self.val_batch_size is not None else '(inherit from loader_params)'}")
        print(f"Cache Train Batches:                       "
              f"{self.cache_train_batches}")

        # build the data dictionary for the loader after data transforms
        base_data = {
            'x': data.x,
            'y': data.y,
            'edge_index': data.edge_index,
            'xy_coordinates': data.xy_coordinates,
            'train_mask': data.train_mask,
            'val_mask': data.val_mask,
            'test_mask': data.test_mask,
            'adata_batch_ids': data.adata_batch_ids,
        }
        # `obs_row_index` (per-cell long tensor: row position INTO the source
        # AnnData's `.obs`) is required to look up arbitrary obs columns at
        # inference time. Older datamodules / blobs that don't have it still
        # work — the inference adata builder treats absence as "skip the
        # full-obs propagation".
        if getattr(data, 'obs_row_index', None) is not None:
            base_data['obs_row_index'] = data.obs_row_index

        # `adata_batch_ids_unseen_mask` (per-cell bool) is True for cells
        # whose batch label wasn't in the train-time densification map
        # (predict-time novel batches). The dual model checks this in
        # forward() and swaps in a mean batch-embedding for those cells
        # so the decoder doesn't apply an arbitrary reference batch's
        # covariate.
        if getattr(data, 'adata_batch_ids_unseen_mask', None) is not None:
            base_data['adata_batch_ids_unseen_mask'] = data.adata_batch_ids_unseen_mask

        # `adata_batch_ids_raw` (per-cell long): the SECTION-level raw
        # adata_batch_id (parsed from uns['batch']) broadcast per cell.
        # Used by predict() to map cells back to their source AnnData
        # without inverting the train-time densification (which is
        # ambiguous when multiple held-out batches share dense=0 via
        # the unknown-label fallback).
        if getattr(data, 'adata_batch_ids_raw', None) is not None:
            base_data['adata_batch_ids_raw'] = data.adata_batch_ids_raw

        optional_data_keys = [
            # 'y_cell_types',
            # 'y_niche_types',
            'encoder_conditions',
            'spatial_prior_features',
            'attr_decoder_conditions',
            'adj_decoder_conditions',
        ]
        
        data_dict_for_loader = {
            **base_data,
            **{attr: getattr(data, attr) for attr in optional_data_keys if getattr(data, attr, None) is not None}
        }
        
        data_for_loader = Data(**data_dict_for_loader)

        # keep all other keys that start with 'y_'
        for key in list(data.keys()):
            if key.startswith('y_'):
                setattr(data_for_loader, key, data[key])

        # call the parent class constructor
        super().__init__(
            data=data_for_loader,
            num_workers=self.num_workers,
            loader=loader_type,
            node_sampler=self.sampler_class,
            **self.loader_params,
            **self.sampler_params
        )

        # Stash the per-AnnData obs DataFrames here so the model's
        # `compute_metrics` (which only has access to `self.trainer.datamodule`)
        # can pass them into `inference_data_dict_to_adata` to write every
        # input obs column onto the inference output AnnData.
        self.obs_per_batch_id = obs_per_batch_id or {}


    def set_custom_loader_class(
            self,
            custom_loader_name: str
        ) -> Callable:
        """
        This function sets the DataLoader class that will be used to load the data when loader_type is `custom`.

        Parameters:
        -----------
        - custom_loader_name: str
            Name of the DataLoader class that will be used to load the data.

        Returns:
        --------
        - Callable
            DataLoader class that will be used to load the data.
        """
        if custom_loader_name == 'NeighborLoader':
            return NeighborLoader
        else:
            raise NotImplementedError(f"{custom_loader_name} not implemented.")


    def set_custom_sampler_class(
            self,
            custom_sampler_name: str
        ) -> Callable:
        """
        This function sets the Sampler class that will be used to sample the subgraphs when loader_type is `custom`.

        Parameters:
        -----------
        - custom_sampler_name: str
            Name of the Sampler class that will be used to sample the subgraphs.

        Returns:
        --------
        - Callable
            Sampler class that will be used to sample the subgraphs.
        """
        if custom_sampler_name == 'NeighborSampler':
            return NeighborSampler
        else:
            raise NotImplementedError(f"{custom_sampler_name} not implemented.")


    def _loader_perf_kwargs(self) -> dict:
        """
        Performance kwargs forwarded to NeighborLoader (and underlying
        torch DataLoader). Centralised so train / val / test / predict
        loaders all get the same tuning:
          - persistent_workers=True keeps the worker pool alive across
            epochs (avoids fork+import cost per epoch). Requires
            num_workers > 0; falls back to False otherwise.
          - pin_memory=True lets the data side overlap H2D copies with
            the GPU forward pass.
          - prefetch_factor=8 lets each worker pre-build several batches
            ahead of the current one. PyG's NeighborSampler is
            CPU-bound, so this hides sampling latency behind GPU work.
            Bumped from 4 -> 8 after wandb showed GPU utilisation
            sitting at 5-10%; doubling the in-flight batches lets the
            workers stay further ahead of the GPU consumer when
            individual sampling calls have variable latency.
        Total expected speedup: 1.3-2x wall-clock on CPU-sampler-heavy
        datasets (mmb20).

        If a caller passed any of these kwargs explicitly via
        `loader_params` (e.g. the run_squint base config), DO NOT
        re-add them here — that would land on
        `NeighborLoader(..., **self.loader_params, **self._loader_perf_kwargs())`
        with each key set twice, raising `TypeError: got multiple
        values for keyword argument '<key>'`. Caller-provided settings
        win silently.
        """
        kw: dict = {"pin_memory": True}
        if int(self.num_workers) > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 8
        # Drop any keys the caller already set in `loader_params` so
        # the eventual unpack at the loader call site can't see
        # duplicates. `getattr` with a default keeps this safe before
        # `self.loader_params` is assigned (e.g. if a subclass calls
        # this helper during its own __init__).
        for k in list(kw):
            if k in getattr(self, "loader_params", {}):
                del kw[k]
        return kw

    def train_dataloader(self):
        """
        This function constructs the DataLoader object for training based on the settings defined in the constructor of the InMemoryDataModule class. If the loader_name is set to 'DefaultFullLoader' or 'DefaultNodeLoader', the function will call the parent class' train_dataloader() function. Otherwise, it will instantiate `self.loader_class` with `self.sampler_class`.

        Notes:
        ------
        - `sample_neighbors_for_inference` is ignored for training.
        - When `cache_train_batches=True`, the heavy NeighborLoader is
          run ONCE to materialise an epoch of `Batch` objects, then a
          lightweight DataLoader (num_workers=0, batch_size=1, identity
          collate) serves them on every subsequent epoch with reshuffled
          ORDER but identical batch CONTENT. See constructor docstring
          for the full trade-off.
        """
        if self.loader_name in ['DefaultFullLoader', 'DefaultNodeLoader']:
            return super().train_dataloader()

        else:
            # Sampler / loader params for THIS train dataloader. Shallow
            # copies so the cache-time `[-1]` override below doesn't
            # leak into `self.sampler_params` (val/test/predict still
            # read those for their own loaders).
            sampler_params = dict(self.sampler_params)
            loader_params  = dict(self.loader_params)

            # When caching is enabled, override num_neighbors to [-1]
            # (full neighborhood) for the cache build.
            #
            # Rationale: the whole point of caching is to remove
            # per-epoch resampling — every epoch sees the same batches.
            # Given that, freezing a RANDOM 8-neighbor subset (the
            # default training fanout) per cell gives strictly less
            # information than freezing the FULL neighborhood. Same
            # staticness either way; full-fanout just stores more
            # signal per cell.
            #
            # Memory cost: spatial k-NN graphs have avg degree ~10-12,
            # so [-1] is ~25-50% bigger per cell than [8]. For
            # mmb-smb scale that's ~2-3 GB cache (vs ~1-2 GB with [8]) —
            # still comfortable inside the 128 GB LSF default. For
            # deeper fanouts like [8, 8] the bump is more like 2-3x
            # because BOTH hops go full; chl59-scale graphs may want
            # `cache_train_batches=False` if memory tightens.
            #
            # The override applies to the CACHE-BUILD loader only —
            # val/test/predict loaders read `self.sampler_params`
            # untouched.
            if self.cache_train_batches:
                original_fanout = sampler_params.get('num_neighbors')
                if original_fanout:
                    n_hops = len(original_fanout)
                    sampler_params['num_neighbors'] = [-1] * n_hops
                    print(
                        f"[InMemoryDataModule] cache_train_batches=True — "
                        f"overriding train num_neighbors from "
                        f"{original_fanout} → "
                        f"{sampler_params['num_neighbors']} for the cache "
                        f"build (full neighborhood is strictly more "
                        f"informative than a frozen random subset; "
                        f"per-epoch resampling is lost either way)."
                    )

            # instantiate the sampler class for training
            train_sampler = self.sampler_class(
                            data=self.data,
                            **sampler_params,
                        )

            # instantiate the loader class for training
            train_loader = self.loader_class(
                                        data=self.data,
                                        num_workers=self.num_workers,
                                        input_nodes=self.input_train_nodes,
                                        neighbor_sampler=train_sampler,
                                        shuffle=False,
                                        **loader_params,
                                        **sampler_params,
                                        **self._loader_perf_kwargs(),
                                    )

            # Legacy / opt-out path: return the NeighborLoader as-is and
            # let workers sample neighborhoods fresh every epoch.
            if not self.cache_train_batches:
                return train_loader

            # Pre-cache path: materialise one full epoch of `Batch`
            # objects into a list (the heavy CPU work happens here, once),
            # then wrap the list in a lightweight DataLoader. Subsequent
            # epochs do zero CPU sampling.
            if self._cached_train_batches is None:
                t0 = time.time()
                print(
                    "[InMemoryDataModule] cache_train_batches=True — "
                    "pre-caching one epoch of training batches (this "
                    "runs the full NeighborSampler path once; "
                    "subsequent epochs serve from CPU memory) ..."
                )
                # Pulling the iterator to exhaustion forces the workers
                # to do their full pass. The workers are torn down when
                # `train_loader`'s iterator is garbage-collected at the
                # end of this scope — we don't keep `train_loader` alive
                # past the cache build.
                #
                # `.cpu()` is critical: some PyG versions return
                # device-mixed Batch objects from NeighborLoader when
                # `pin_memory=True` is combined with CUDA-aware pyg-lib
                # (a stray CUDA tensor inside an otherwise-CPU batch).
                # If we cache those as-is, the outer DataLoader's
                # `pin_memory=True` step downstream blows up with
                # `cannot pin 'torch.cuda.FloatTensor' only dense CPU
                # tensors can be pinned`. Calling `.cpu()` here is a
                # no-op on already-CPU tensors and a forced device move
                # on any GPU tensor, so the cache is guaranteed to be
                # all-CPU regardless of the inner loader's behaviour.
                cached: list = []
                for batch in train_loader:
                    cached.append(batch.cpu())
                self._cached_train_batches = cached
                dt = time.time() - t0
                n_batches = len(self._cached_train_batches)
                print(
                    f"[InMemoryDataModule] cached {n_batches} train "
                    f"batches in {dt:.1f}s. Subsequent epochs will "
                    f"reuse this list (num_workers=0, identity collate)."
                )

            # Lightweight DataLoader over the cached list. Batch order
            # reshuffles every epoch (preserves a bit of stochasticity);
            # batch CONTENT is fixed by the cache.
            #
            # `pin_memory=False` on purpose: the cached `Batch` objects
            # are heterogeneous PyG containers, and PyTorch's per-fetch
            # `pin_memory` step recurses through `Batch.pin_memory()` →
            # one stray non-pinnable tensor (sparse, GPU, custom) and
            # the whole step crashes. The H2D-transfer cost we'd save
            # by pinning is marginal anyway (small batches, 80-epoch
            # reuse — the OS page cache absorbs most of it). Lightning's
            # `transfer_batch_to_device` still moves each batch to GPU
            # before the training step; we just don't pin in advance.
            cached_loader = DataLoader(
                self._cached_train_batches,
                batch_size=1,
                shuffle=True,
                collate_fn=_identity_collate,
                num_workers=0,        # no sampling needed any more
                pin_memory=False,     # see comment above — avoid pin-crash
            )
            return cached_loader


    def val_dataloader(self):
        """
        This function constructs the DataLoader object for validation based on the settings defined in the constructor of the InMemoryDataModule class. If the loader_name is set to 'DefaultFullLoader' or 'DefaultNodeLoader', the function will call the parent class' val_dataloader() function. Otherwise, it will instantiate `self.loader_class` with `self.sampler_class`.

        Notes:
        ------
        - If `sample_neighbors_for_inference` is set to True, the neighbors are sampled during inference.
        - If `sample_neighbors_for_inference` is set to False, the neighbors are not sampled during inference.
        - `val_dataloader` loops over nodes in the validation set.
        """
        if self.loader_name in ['DefaultFullLoader', 'DefaultNodeLoader']:
            return super().val_dataloader()

        else:
            # Shallow copy so the `[-1]` override below doesn't mutate
            # `self.sampler_params` and leak into other dataloaders
            # (predict can have a different fanout from val/test now).
            sampler_params = dict(self.sampler_params)

            if not self.sample_neighbors_for_inference:
                # update num_neighbors to -1 so that the neighbors are not sampled.
                sampler_params['num_neighbors'] = [-1] * len(self.sampler_params['num_neighbors'])

            # Same shallow-copy trick for loader_params: if
            # val_batch_size is set, we override `batch_size` only for
            # the val loader, leaving training + predict untouched.
            loader_params = dict(self.loader_params)
            if self.val_batch_size is not None:
                loader_params["batch_size"] = self.val_batch_size

            # instantiate the sampler class for validation
            val_sampler = self.sampler_class(
                                data=self.data,
                                **sampler_params,
                            )

            # instantiate the loader class for validation
            val_loader = self.loader_class(
                                data=self.data,
                                num_workers=self.num_workers,
                                input_nodes=self.input_val_nodes,
                                neighbor_sampler=val_sampler,
                                shuffle=False,
                                **loader_params,
                                **sampler_params,
                                **self._loader_perf_kwargs(),
                            )
            return val_loader


    def test_dataloader(self):
        """
        This function constructs the DataLoader object for testing based on the settings defined in the constructor of the InMemoryDataModule class. If the loader_name is set to 'DefaultFullLoader' or 'DefaultNodeLoader', the function will call the parent class' test_dataloader() function. Otherwise, it will instantiate `self.loader_class` with `self.sampler_class`.

        Notes:
        ------
        - If `sample_neighbors_for_inference` is set to True, the neighbors are sampled during inference.
        - If `sample_neighbors_for_inference` is set to False, the neighbors are not sampled during inference.
        - `test_dataloader` loops over nodes in the test set.
        - `val_batch_size` (when set) ALSO applies to the test loader.
          They share the same no-grad / lighter-fanout characteristics
          so the same batch-size override usually makes sense for both.
        """
        if self.loader_name in ['DefaultFullLoader', 'DefaultNodeLoader']:
            return super().test_dataloader()

        else:
            # Shallow copy so the `[-1]` override below doesn't mutate
            # `self.sampler_params` and leak into other dataloaders.
            sampler_params = dict(self.sampler_params)

            if not self.sample_neighbors_for_inference:
                # update num_neighbors to -1 so that the neighbors are not sampled.
                sampler_params['num_neighbors'] = [-1] * len(self.sampler_params['num_neighbors'])

            # Same shallow-copy trick for loader_params (see val_dataloader
            # above). The `val_batch_size` knob applies to both val and
            # test — they share the same usage pattern (no-grad pass
            # over a held-out subset with lighter fanout than predict).
            loader_params = dict(self.loader_params)
            if self.val_batch_size is not None:
                loader_params["batch_size"] = self.val_batch_size

            # instantiate the sampler class for testing
            test_sampler = self.sampler_class(
                                data=self.data,
                                **sampler_params,
                            )

            # instantiate the loader class for testing
            test_loader = self.loader_class(
                                data=self.data,
                                num_workers=self.num_workers,
                                input_nodes=self.input_test_nodes,
                                neighbor_sampler=test_sampler,
                                shuffle=False,
                                **loader_params,
                                **sampler_params,
                                **self._loader_perf_kwargs(),
                            )
            return test_loader


    def predict_dataloader(self):
        """
        This method returns a DataLoader object for all the nodes in the graph.
        This is required for looping over all the nodes in the graph for inference.

        Notes:
        ------
        - The PREDICT path consults `predict_sample_neighbors_for_inference`
          (NOT `sample_neighbors_for_inference` — that's for val/test).
          Default is `False` → fans out with `num_neighbors=[-1]` (every
          neighbor) so the embeddings written to AnnData are
          deterministic and aggregate the full neighborhood. Set the
          flag to `True` if you want stochastic predict-time sampling.
        - The PREDICT path can also use a separate `predict_batch_size`
          (set on the datamodule). Default is "inherit
          `loader_params['batch_size']`" — same as legacy. Bumping it
          (e.g. 1024 with a 256-train-batch run) reduces the total
          number of predict batches without changing model outputs;
          training-time batch_size is tuned for gradient dynamics and
          isn't relevant during a no-grad predict pass.
        - `predict_dataloader` loops over all the nodes in the graph.
        """
        # Shallow copy so the `[-1]` override below doesn't mutate
        # `self.sampler_params` (val/test may want a different fanout).
        sampler_params = dict(self.sampler_params)

        # update num_neighbors to -1 so that the neighbors are not sampled.
        if not self.predict_sample_neighbors_for_inference:
            sampler_params['num_neighbors'] = [-1] * len(self.sampler_params['num_neighbors'])

        # Same shallow-copy logic for loader_params: if predict_batch_size
        # is set, we need to override `batch_size` for THIS loader only,
        # without mutating self.loader_params (which val/test/train share).
        loader_params = dict(self.loader_params)
        if self.predict_batch_size is not None:
            loader_params["batch_size"] = self.predict_batch_size

        # instantiate the sampler class to control the behavior of the loader
        infer_sampler = self.sampler_class(
                            data=self.data,
                            **sampler_params,
                        )

        # instantiate the loader class for inference
        # input_nodes is set to None. That is, all nodes are used.
        infer_loader = self.loader_class(
                                    data=self.data,
                                    num_workers=self.num_workers,
                                    input_nodes=None,
                                    neighbor_sampler=infer_sampler,
                                    shuffle=False,
                                    **loader_params,
                                    **sampler_params,
                                    **self._loader_perf_kwargs(),
                                )
        return infer_loader
