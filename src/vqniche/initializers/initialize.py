import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Literal

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
    conditioning_sources = config['dataset'].get(
                            'conditioning_sources',
                            [],
                        )
    ExperimentDataKeys = SetExperimentDataKeys(
                            feature_names=feature_names,
                            label_name=label_name,
                            edge_index_name=edge_index_name,
                            conditioning_sources=conditioning_sources
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
    data_batch.num_features = int(data_batch.num_features)
    data_batch.num_classes = int(data_batch.num_classes)
    data_batch.condition_dim = int(data_batch.condition_dim)

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
        condition_dim: int,
        out_channels: int,
    ) -> pl.LightningModule:
    # set model class
    Model = set_model_class(model_name=config['model']['model_name'])

    # get model, optimizer, loss, and task parameters
    model_name = config['model']['model_name']
    encoder_name = config['model']['encoder_name']
    attribute_decoder_name = config['model']['attribute_decoder_name']
    adjacency_decoder_name = config['model']['adjacency_decoder_name']
    predictor_name = config['model']['predictor_name']
    train_metrics_list = config['model']['train_metrics_list']
    encoder_params = config['model']['encoder_params']
    attribute_decoder_params = config['model']['attribute_decoder_params']
    adjacency_decoder_params = config['model']['adjacency_decoder_params']
    optimizer_params = config['model']['optimizer_params']
    loss_params = config['model']['loss_params']

    # initialize model
    model = Model(
                model_name=model_name,
                encoder_name=encoder_name,
                attribute_decoder_name=attribute_decoder_name,
                adjacency_decoder_name=adjacency_decoder_name,
                predictor_name=predictor_name,
                in_channels=in_channels,
                condition_dim=condition_dim,
                out_channels=out_channels,
                encoder_params=encoder_params,
                train_metrics_list=train_metrics_list,
                attribute_decoder_params=attribute_decoder_params,
                adjacency_decoder_params=adjacency_decoder_params,
                optimizer_params=optimizer_params,
                loss_params=loss_params,
            )
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