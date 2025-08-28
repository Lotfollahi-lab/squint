import os
import re
import yaml
from pathlib import Path
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
        cell_ids: List[List[str]], max_batch: int = 38,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a nested list of cell_ids, return a tuple of tensors containing
    batch IDs and one-hot encodings of batch IDs (0..max_batch).
    
    Parameters
    ----------
    cell_ids : List[List[str]]
        Nested list of cell identifiers, e.g. [['1_batch1_0', '1_batch1_1'], ...]
        num_cells = \sum_{i=1}^{len(cell_ids)} |cell_ids[i]|
    max_batch : int
        Maximum batch index (default 38 → makes one-hot vectors of length 39).
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - Batch ID tensor of shape (num_cells,)
        - One-hot tensor of shape (num_cells, max_batch + 1)
    """
    batch_ids = []
    batch_one_hot = []
    for group in cell_ids:
        group_ids = []
        group_tensor = []
        for cid in group:
            # extract the batch number from pattern "batchX"
            match = re.search(r"batch(\d+)", cid)
            if not match:
                raise ValueError(f"Could not parse batch from id: {cid}")
            b = int(match.group(1))
            
            one_hot = torch.zeros(max_batch + 1, dtype=torch.float)
            one_hot[b] = 1.0
            group_ids.append(b)
            group_tensor.append(one_hot)
        batch_ids.extend(group_ids)
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
    user_config_path = Path(logger.experiment.dir) / 'user_specified_config.yaml'
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
    attr_decoder_condition_list = config['model']['attribute_decoder_params'].get(
                                    'conditioning_params',
                                    {},
                                ).get(
                                    'condition_list',
                                    None,
                                )
    adj_decoder_condition_list = config['model']['adjacency_decoder_params'].get(
                                    'conditioning_params',
                                    {},
                                ).get(
                                    'condition_list',
                                    None,
                                )
    
    ExperimentDataKeys = SetExperimentDataKeys(
                            feature_names=feature_names,
                            label_name=label_name,
                            edge_index_name=edge_index_name,
                            encoder_condition_list=encoder_condition_list,
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
    attr_decoder_condition_list = config['model']['attribute_decoder_params'].get(
                                    'conditioning_params',
                                    {},
                                ).get(
                                    'condition_list',
                                    None,
                                )

    # TODO: clean up this multi-section conditioning code
    batch_ids, batch_conditions = build_batch_one_hot(
                                            cell_ids=data_batch.cell_id,
                                            max_batch=len(dataset_blob),
                                        )
    data_batch.adata_batch_ids = batch_ids

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
    model_param_dict = {
        'model_name': model_name,
        'encoder_name': config['model']['encoder_name'],
        'attribute_decoder_name': config['model']['attribute_decoder_name'],
        'adjacency_decoder_name': config['model']['adjacency_decoder_name'],
        'predictor_name': config['model']['predictor_name'],
        'train_metrics_list': config['model']['train_metrics_list'],
        'in_channels': in_channels,
        'out_channels': out_channels,
        'encoder_params': config['model']['encoder_params'],
        'attribute_decoder_params': config['model']['attribute_decoder_params'],
        'adjacency_decoder_params': config['model']['adjacency_decoder_params'],
        'optimizer_params': config['model']['optimizer_params'],
        'loss_params': config['model']['loss_params'],
    }

    if model_name == 'VQNiche':
        model_param_dict['imputation_params'] = config['model']['imputation_params']

    # --------------------- Initialize Model ---------------------
    Model = set_model_class(model_name=model_name)
    model = Model(**model_param_dict)
    return model


def set_wandb_experiment_dir(
        config: Dict,
        experiment_mode: Literal['sweep', 'standalone'] = 'standalone',
        sweep_id: Optional[str] = None,
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
        exp_dir = exp_dir / f"sweep-{sweep_id}"

    # create experiment run directory
    exp_dir.mkdir(parents=True, exist_ok=True)

    # set environment variable for wandb
    os.environ["WANDB_DIR"] = str(exp_dir)

    return exp_dir