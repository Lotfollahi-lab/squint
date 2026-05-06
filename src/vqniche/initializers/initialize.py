import os
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Literal, List, Tuple

import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader as BatchBuilder
from pytorch_lightning.loggers import WandbLogger

from ..preprocessors.graph_constructors import set_edge_index_name
from ..dataset.transforms import SetExperimentDataKeys, init_gene_count_transforms, init_train_transforms
from ..dataset.in_memory_dataset_blob import InMemoryDatasetBlob
from ..dataloaders.in_memory_datamodule import InMemoryDataModule
from ..models.vanilla_mlp import VanillaMLP
from ..models.vanilla_gnn import VanillaGNN
from ..models.vqniche import VQNiche
from .utils import safe_int_conversion


def build_batch_one_hot(
        cell_ids: List[List[str]],
        max_batch: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a nested list of cell_ids, return dense batch IDs + their one-hot encoding.

    The function inspects every cell_id, parses its raw batch number out of the
    "batchN" substring, and remaps the *unique* raw numbers to dense indices
    [0, num_unique_batches) in sorted order.  The returned batch ID tensor and
    one-hot encoding both use the dense indices, so the one-hot dimension equals
    the number of distinct batches actually present in `cell_ids` — not the
    maximum raw batch number.

    Example: raw batch numbers {15, 82} are remapped to dense {0, 1}; one-hot
    shape becomes (num_cells, 2) instead of (num_cells, 83).

    Parameters
    ----------
    cell_ids : List[List[str]]
        Nested list of cell identifiers, e.g. [['1_batch1_0', '1_batch1_1'], ...]
        num_cells = \\sum_{i=1}^{len(cell_ids)} |cell_ids[i]|
    max_batch : int, optional
        Deprecated. Kept for backward compatibility; ignored.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - Dense batch ID tensor of shape (num_cells,), values in [0, n_unique_batches).
        - One-hot tensor of shape (num_cells, n_unique_batches).
    """
    # Pass 1: collect raw batch numbers, build raw -> dense mapping.
    raw_per_cell: list[int] = []
    for group in cell_ids:
        for cid in group:
            match = re.search(r"batch(\d+)", cid)
            if not match:
                raise ValueError(f"Could not parse batch from id: {cid}")
            raw_per_cell.append(int(match.group(1)))

    unique_raw = sorted(set(raw_per_cell))
    raw_to_dense = {r: i for i, r in enumerate(unique_raw)}
    n_classes = len(unique_raw)

    # Pass 2: densify and build one-hot. Pass 1 already validated that every
    # cid contains "batchN", so the regex below cannot match None — but we
    # guard explicitly to fail loudly if Pass 1's invariants are broken.
    batch_ids = []
    batch_one_hot = []
    for group in cell_ids:
        group_tensor = []
        for cid in group:
            match = re.search(r"batch(\d+)", cid)
            if not match:
                raise ValueError(f"Could not parse batch from id: {cid}")
            r = int(match.group(1))
            d = raw_to_dense[r]
            one_hot = torch.zeros(n_classes, dtype=torch.float)
            one_hot[d] = 1.0
            batch_ids.append(d)
            group_tensor.append(one_hot)
        batch_one_hot.append(torch.stack(group_tensor))

    batch_ids = torch.tensor(batch_ids, dtype=torch.long)
    batch_one_hot = torch.cat(batch_one_hot, dim=0)

    return batch_ids, batch_one_hot


def build_batch_one_hot_from_obs(
        obs_batch: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a per-cell batch one-hot from `adata.obs[batch_key]` values that
    have been collected as per-AnnData lists in the dataset blob.

    Unlike `build_batch_one_hot`, this does NOT parse anything out of
    `cell_id` — it directly densifies the raw obs labels (strings or ints)
    to [0, n_unique_labels) in sorted order. Use this when the AnnData
    files have an explicit `obs[batch_key]` column instead of (or in
    addition to) batch info encoded in `cell_id` strings.

    Parameters
    ----------
    obs_batch : List[List[<str|int>]]
        Nested list of obs[batch_key] values, one inner list per AnnData
        batch in the dataset blob. Inner-list element types may be str or
        int — both are normalised to str before lookup so 'Lung5_Rep1'
        and 1 are both supported.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Dense batch ID tensor of shape (num_cells,), values in
          [0, n_unique_labels).
        - One-hot tensor of shape (num_cells, n_unique_labels).
    """
    # Pass 1: collect unique labels, build label -> dense map.
    raw_per_cell: list[str] = []
    for group in obs_batch:
        for label in group:
            raw_per_cell.append(str(label))

    unique_labels = sorted(set(raw_per_cell))
    label_to_dense = {lbl: i for i, lbl in enumerate(unique_labels)}
    n_classes = len(unique_labels)

    # Pass 2: densify + one-hot.
    batch_ids = []
    batch_one_hot = []
    for group in obs_batch:
        group_tensor = []
        for label in group:
            d = label_to_dense[str(label)]
            one_hot = torch.zeros(n_classes, dtype=torch.float)
            one_hot[d] = 1.0
            batch_ids.append(d)
            group_tensor.append(one_hot)
        batch_one_hot.append(torch.stack(group_tensor))

    batch_ids = torch.tensor(batch_ids, dtype=torch.long)
    batch_one_hot = torch.cat(batch_one_hot, dim=0)
    return batch_ids, batch_one_hot


def build_timepoint_one_hot(
        batch_ids: torch.Tensor,
        max_timepoint: int = 4,
        batch_timepoint_map: Dict[int, int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor of batch IDs, return a tuple of tensors containing
    timepoint IDs and one-hot encodings of timepoint IDs (0..max_timepoint).

    Parameters
    ----------
    batch_ids : torch.Tensor
        Tensor of batch IDs
    max_timepoint : int
        Maximum timepoint index (default 4 → makes one-hot vectors of length 5).
    batch_timepoint_map : Dict[int, int]
        Dictionary mapping batch IDs to timepoint IDs.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - Timepoint ID tensor of shape (num_cells,)
        - One-hot tensor of shape (num_cells, max_timepoint)
    """
    timepoint_ids = []
    timepoint_one_hot = []
    for batch_id in batch_ids:
        timepoint_id = batch_timepoint_map[batch_id.item()]
        timepoint_ids.append(timepoint_id)
        timepoint_one_hot.append(torch.zeros(max_timepoint, dtype=torch.float))
        timepoint_one_hot[-1][timepoint_id] = 1.0
    timepoint_ids = torch.tensor(timepoint_ids, dtype=torch.long)
    timepoint_one_hot = torch.stack(timepoint_one_hot)
    return timepoint_ids, timepoint_one_hot


def initialize_logger(
        config: Dict,
    ) -> WandbLogger:

    logger = WandbLogger(
                log_model=config['logging']['log_model'],
            )

    # Save the complete original user-specified configuration
    try:
        user_config_path = Path(logger.experiment.dir) / 'user_specified_config.yaml'
    except:
        # TODO: when running a wandb sweep, logger.experiment.dir is a method for some reason
        # the rest of the sweep runs correctly, but the start fails weirdly. 
        # needs investigation.
        print(logger.experiment.dir)
        print(type(logger.experiment.dir))

    with open(user_config_path, 'w') as config_file:
        yaml.dump(config, config_file)

    return logger


def initialize_dataset_blob(
        config: Dict,
    ) -> InMemoryDatasetBlob:
    # --------------------- Initialize Transforms ---------------------
    # 1. gene count transforms: e.g. subset HVGs, normalize features, etc.
    gene_count_transform_names = config['dataset']['gene_count_transform_names']
    gene_count_transform_params = config['dataset']['gene_count_transform_params']
    GeneCountTransforms = init_gene_count_transforms(
                        gene_count_transform_names=gene_count_transform_names,
                        **gene_count_transform_params
                    )

    # 2. set experiment data keys: e.g. feature names, label name, edge index name
    graph_params = config['dataset']['graph_params']
    edge_index_name = set_edge_index_name(
                        spatial_key=graph_params['spatial_key'],
                        delaunay=graph_params['delaunay'],
                        n_neighs=graph_params['n_neighs'],
                        radius=graph_params['radius'],
                    )
    feature_names = config['dataset']['feature_names']
    label_name = config['dataset']['label_name']
    
    encoder_condition_list = config['model']['encoder_params'].get(
                                'conditioning_params',
                                {},
                            ).get(
                                'condition_list',
                                None,
                            )
    
    spatial_prior_feature = config['model']['encoder_params'].get(
                                    'spatial_prior_params',
                                    {},
                                ).get(
                                    'spatial_prior_feature',
                                    None,
                                )
    
    # Decoder-conditioning lookup. Single-codebook configs (VQNiche etc.)
    # carry `attribute_decoder_params` / `adjacency_decoder_params`. The
    # dual-codebook config (VQNiche_Dual) replaces them with per-branch
    # equivalents and removes the adjacency decoder entirely. Be tolerant
    # of either layout.
    def _get_condition_list(decoder_params: dict) -> Optional[list]:
        return decoder_params.get('conditioning_params', {}).get('condition_list', None)

    if 'attribute_decoder_params' in config['model']:
        attr_decoder_condition_list = _get_condition_list(
            config['model']['attribute_decoder_params']
        )
    else:
        # Dual model: take the union of conditioning lists across the cell
        # and niche decoders (they typically share conditioning, but be
        # defensive). Order is preserved relative to the first occurrence.
        cond_lists = []
        for branch_key in ('attribute_decoder_cell_params',
                           'attribute_decoder_niche_params'):
            branch_params = config['model'].get(branch_key, {})
            cl = _get_condition_list(branch_params)
            if cl:
                cond_lists.append(cl)
        if not cond_lists:
            attr_decoder_condition_list = None
        else:
            seen = set()
            attr_decoder_condition_list = []
            for cl in cond_lists:
                for c in cl:
                    if c not in seen:
                        seen.add(c)
                        attr_decoder_condition_list.append(c)

    if 'adjacency_decoder_params' in config['model']:
        adj_decoder_condition_list = _get_condition_list(
            config['model']['adjacency_decoder_params']
        )
    else:
        # Dual model has no MLP adjacency decoder — no conditioning to fetch.
        adj_decoder_condition_list = None
    
    ExperimentDataKeys = SetExperimentDataKeys(
                            feature_names=feature_names,
                            label_name=label_name,
                            edge_index_name=edge_index_name,
                            encoder_condition_list=encoder_condition_list,
                            spatial_prior_feature=spatial_prior_feature,
                            attr_decoder_condition_list=attr_decoder_condition_list,
                            adj_decoder_condition_list=adj_decoder_condition_list,
                        )

    # 3. train transforms: e.g. random node split, etc.
    train_transform_names = config['dataset']['train_transform_names']
    train_transform_params = config['dataset']['train_transform_params']
    TrainTransforms = init_train_transforms(
                        train_transform_names=train_transform_names,
                        **train_transform_params
                    )

    # initialize a composed transform
    # NOTE: transforms are not order-invariant
    transforms_list = GeneCountTransforms + [ExperimentDataKeys] + TrainTransforms
    transforms = T.Compose(transforms_list)

    # --------------------- Initialize Dataset Blob ---------------------
    # set root data directory
    root_data_dir = config['dataset']['root_data_dir']
    dataset_name = config['dataset']['dataset_name']

    # initialize pytorch geometric dataset blob stored at:
    # root_data_dir / 'gold' / 'in-memory-PyG-dataset-blob' / dataset_name / 'dataset_blob.pt'
    dataset_blob = InMemoryDatasetBlob(
                        name=dataset_name,
                        data_directory_path=root_data_dir,
                        transform=transforms
                    )

    return dataset_blob


def initialize_databatch(
        config: Dict,
        dataset_blob: InMemoryDatasetBlob,
    ) -> Batch:
    # load PyG data object(s) corresponding to adata_batch_idx (e.g. 0 -> AnnData batch0)
    # NOTE: sss2-1b_1p is 1-indexed, while others are 0-indexed
    adata_batch_idx = config['dataset']['adata_batch_idx']

    # list of Data objects, one for each tissue section
    if isinstance(adata_batch_idx, int):
        # -1 means use all batches
        if adata_batch_idx == -1:
            adata_batch_idx = list(range(len(dataset_blob)))
        else:
            adata_batch_idx = [adata_batch_idx]
    data_list = [dataset_blob[idx] for idx in adata_batch_idx]

    # collate the list of Data objects into a single Batch object
    # i.e. concatenate tissue sections into one big graph with disconnected components
    data_batch = BatchBuilder(
                        dataset=data_list,
                        batch_size=len(data_list),
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                    ).collate_fn(data_list)
    data_batch.adata_batch_id = torch.tensor(
        [int(d.adata_batch_id if isinstance(d.adata_batch_id, int)
              else d.adata_batch_id.view(-1)[0].item())
         for d in data_list],
        dtype=torch.long
    )    

    # TODO: fix this hard-coding
    data_batch.num_features = safe_int_conversion(data_batch.num_features)
    data_batch.num_classes = safe_int_conversion(data_batch.num_classes)

    # --------------------- Set Section-Level Conditioning Features ---------------------
    encoder_condition_list = config['model']['encoder_params'].get(
                                'conditioning_params',
                                {},
                            ).get(
                                'condition_list',
                                None,
                            )
    # Decoder-conditioning lookup is dual-config-aware: fall back to
    # `attribute_decoder_cell_params` + `attribute_decoder_niche_params`
    # (taking the union of their condition lists) when the legacy
    # single-decoder key is absent (VQNiche_Dual layout).
    if 'attribute_decoder_params' in config['model']:
        attr_decoder_condition_list = config['model']['attribute_decoder_params'].get(
                                        'conditioning_params',
                                        {},
                                    ).get(
                                        'condition_list',
                                        None,
                                    )
    else:
        cond_lists = []
        for branch_key in ('attribute_decoder_cell_params',
                           'attribute_decoder_niche_params'):
            branch_params = config['model'].get(branch_key, {})
            cl = branch_params.get('conditioning_params', {}).get('condition_list', None)
            if cl:
                cond_lists.append(cl)
        if not cond_lists:
            attr_decoder_condition_list = None
        else:
            seen = set()
            attr_decoder_condition_list = []
            for cl in cond_lists:
                for c in cl:
                    if c not in seen:
                        seen.add(c)
                        attr_decoder_condition_list.append(c)

    # TODO: clean up this multi-section conditioning code
    # Two sources for the per-cell batch one-hot are supported, in priority:
    #
    #   1. `data_batch.obs_batch` — the per-cell `adata.obs[batch_key]`
    #      values collected during dataset-blob construction. Densifies
    #      unique label strings to contiguous indices.
    #   2. `data_batch.cell_id` — legacy fallback that parses "batchN"
    #      substrings out of cell_id strings. Used when the AnnDatas
    #      didn't have an `obs[batch_key]` column at blob-build time.
    #
    # `build_batch_one_hot[*]` densifies the raw labels to contiguous
    # indices [0, n_unique_batches), so `max_batch` is no longer needed
    # and the one-hot dim matches the number of distinct samples.
    if hasattr(data_batch, 'obs_batch') and data_batch.obs_batch is not None:
        batch_ids, batch_conditions = build_batch_one_hot_from_obs(
            obs_batch=data_batch.obs_batch,
        )
    else:
        batch_ids, batch_conditions = build_batch_one_hot(
            cell_ids=data_batch.cell_id,
        )
    data_batch.adata_batch_ids = batch_ids

    if encoder_condition_list is not None:
        if 'cell_batch_id' in encoder_condition_list:
            data_batch.encoder_conditions = torch.cat(
                                                [data_batch.encoder_conditions, batch_conditions],
                                                dim=-1,
                                            )
            data_batch.encoder_condition_dim = data_batch.encoder_conditions.shape[1]
        if 'timepoint_id' in encoder_condition_list:
            _, timepoint_conditions = build_timepoint_one_hot(
                                            batch_ids=batch_ids,
                                            **config['dataset']['batch_timepoint'],
                                        )
            data_batch.encoder_conditions = torch.cat(
                                                    [data_batch.encoder_conditions, timepoint_conditions],
                                                    dim=-1,
                                                )
            data_batch.encoder_condition_dim = data_batch.encoder_conditions.shape[1]

    if attr_decoder_condition_list is not None:
        if 'cell_batch_id' in attr_decoder_condition_list:
            data_batch.attr_decoder_conditions = torch.cat(
                                                    [data_batch.attr_decoder_conditions, batch_conditions],
                                                    dim=-1,
                                                )
            data_batch.attr_decoder_condition_dim = data_batch.attr_decoder_conditions.shape[1]
        if 'timepoint_id' in attr_decoder_condition_list:
            _, timepoint_conditions = build_timepoint_one_hot(
                                            batch_ids=batch_ids,
                                            **config['dataset']['batch_timepoint'],
                                        )
            data_batch.attr_decoder_conditions = torch.cat(
                                                    [data_batch.attr_decoder_conditions, timepoint_conditions],
                                                    dim=-1,
                                                )
            data_batch.attr_decoder_condition_dim = data_batch.attr_decoder_conditions.shape[1]

    data_batch.encoder_condition_dim = safe_int_conversion(data_batch.encoder_condition_dim)
    data_batch.spatial_prior_feature_dim = safe_int_conversion(data_batch.spatial_prior_feature_dim)
    data_batch.attr_decoder_condition_dim = safe_int_conversion(data_batch.attr_decoder_condition_dim)
    data_batch.adj_decoder_condition_dim = safe_int_conversion(data_batch.adj_decoder_condition_dim)

    # --------------------- Print Data Batch ---------------------
    print(data_batch)

    print(f"Batch ID(s): {adata_batch_idx}")
    print(f"Data Batch: {data_batch}")
    print(f"Number of Tissue Sections: {len(data_list)}")

    return data_batch


def initialize_datamodule(
        config: Dict,
        data: Data,
    ) -> pl.LightningDataModule:
    # set parameters for data loader and sampler for training, validation, and testing
    loader_name = config['datamodule']['loader_name']
    loader_params = config['datamodule']['loader_params']

    sampler_name = config['datamodule']['sampler_name']
    sampler_params = config['datamodule']['sampler_params']

    inference_params = config['datamodule']['inference_params']

    datamodule_batch = InMemoryDataModule(
                            data=data,
                            loader_name=loader_name,
                            loader_params=loader_params,
                            sampler_name=sampler_name,
                            sampler_params=sampler_params,
                            **inference_params,
                        )
    return datamodule_batch


def set_model_class(
        model_name: str,
    ) -> pl.LightningModule:
    if model_name == 'VanillaMLP':
        Model = VanillaMLP
    elif model_name in ['GraphSAGE', 'GATv2', 'GIN']:
        Model = VanillaGNN
    elif model_name == 'VQNiche':
        Model = VQNiche
    elif model_name == 'VQNiche_Dual':
        from vqniche.models import VQNiche_Dual
        Model = VQNiche_Dual
    else:
        raise ValueError(f"Model {model_name} not found.")
    return Model


def initialize_model(
        config: Dict,
        in_channels: int,
        out_channels: int,
    ) -> pl.LightningModule:
    # --------------------- Set Model Parameters ---------------------
    model_name = config['model']['model_name']

    # Common parameter set used by all models. The dual model accepts the
    # same legacy keys (it just ignores `adjacency_decoder_params`) so we
    # keep one shared dict and extend it model-specifically below.
    model_param_dict = {
        'model_name': model_name,
        'encoder_name': config['model']['encoder_name'],
        'attribute_decoder_name': config['model']['attribute_decoder_name'],
        'adjacency_decoder_name': config['model'].get('adjacency_decoder_name'),
        'predictor_name': config['model']['predictor_name'],
        'train_metrics_list': config['model']['train_metrics_list'],
        'test_metrics_list': config['model']['test_metrics_list'],
        'in_channels': in_channels,
        'out_channels': out_channels,
        'encoder_params': config['model']['encoder_params'],
        'optimizer_params': config['model']['optimizer_params'],
        'loss_params': config['model']['loss_params'],
    }

    # Single-decoder models: use the legacy `attribute_decoder_params` key.
    if model_name in ('VQNiche', 'VanillaMLP', 'GraphSAGE', 'GATv2', 'GIN'):
        model_param_dict['attribute_decoder_params'] = config['model'].get('attribute_decoder_params', {})
        model_param_dict['adjacency_decoder_params'] = config['model'].get('adjacency_decoder_params', {})
    # Dual-decoder model: use the two cell/niche keys (and ignore the
    # legacy single-decoder keys if they happen to be present).
    elif model_name == 'VQNiche_Dual':
        model_param_dict['attribute_decoder_cell_params']  = config['model']['attribute_decoder_cell_params']
        model_param_dict['attribute_decoder_niche_params'] = config['model']['attribute_decoder_niche_params']
        # NicheCompass-style decoder covariate. Set in `train()` after data
        # load (n_unique_batches). Default 0 = off.
        model_param_dict['decoder_covariate_dim'] = int(
            config['model'].get('decoder_covariate_dim', 0)
        )
        # Domain-adversarial batch-invariance head. Set in `train()` after
        # data load (n_unique_batches). Default 0 = off.
        model_param_dict['adversarial_batch_dim'] = int(
            config['model'].get('adversarial_batch_dim', 0)
        )
        model_param_dict['adversarial_alpha'] = float(
            config['model'].get('adversarial_alpha', 1.0)
        )
        adv_hidden = config['model'].get('adversarial_hidden_channels')
        if adv_hidden is not None:
            model_param_dict['adversarial_hidden_channels'] = list(adv_hidden)

    if model_name in ('VQNiche', 'VQNiche_Dual'):
        # Both VQNiche variants accept imputation_params (VQNiche_Dual
        # currently ignores it but accepts the kwarg for forward-compat).
        model_param_dict['imputation_params'] = config['model'].get('imputation_params')

    # --------------------- Initialize Model ---------------------
    Model = set_model_class(model_name=model_name)
    model = Model(**model_param_dict)
    return model


def set_wandb_experiment_dir(
        config: Dict,
        experiment_mode: Literal['sweep', 'standalone'] = 'standalone',
        sweep_name: Optional[str] = None,
    ) -> Path:
    # set root sweep directory
    exp_dir = Path(config['logging']['root_log_dir']) / config['dataset']['dataset_name'] / experiment_mode

    # create model subdirectory
    exp_dir = exp_dir / config['model']['model_name']

    # create batch subdirectory
    exp_dir = exp_dir / f"batch={config['dataset']['adata_batch_idx']}"

    # create edge index subdirectory
    edge_index_name = set_edge_index_name(
                        spatial_key=config['dataset']['graph_params']['spatial_key'],
                        delaunay=config['dataset']['graph_params']['delaunay'],
                        n_neighs=config['dataset']['graph_params']['n_neighs'],
                        radius=config['dataset']['graph_params']['radius'],
                    )
    exp_dir = exp_dir / edge_index_name

    # set experiment run directory
    if experiment_mode == 'sweep':
        assert sweep_name is not None, "Sweep name is required for sweep mode."
        today = datetime.now().strftime('%Y%m%d')
        now = datetime.now().strftime('%H%M%S')
        exp_dir = exp_dir / sweep_name / f"{today}-{now}"

    # create experiment run directory
    exp_dir.mkdir(parents=True, exist_ok=True)

    # set environment variable for wandb
    os.environ["WANDB_DIR"] = str(exp_dir)

    return exp_dir