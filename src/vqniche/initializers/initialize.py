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
from ..dataset.transforms import SetExperimentDataKeys, init_data_transforms
from ..dataset.in_memory_dataset_blob import InMemoryDatasetBlob
from ..dataloaders.in_memory_datamodule import InMemoryDataModule
from ..models.mlp import MLP
from ..models.vanilla_gnn import VanillaGNN
from ..models.vqgraph import VQGraph


def initialize_logger(
        config: Dict,
    ) -> WandbLogger:

    logger = WandbLogger(
                log_model=config['logging']['log_model'],
            )

    config_path = Path(logger.experiment.dir) / 'config.yaml'
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)

    return logger


def initialize_dataset_blob(
        config: Dict,
    ) -> InMemoryDatasetBlob:
    # --------------------- Set Data Keys ---------------------
    # decide which node features to use in this experiment
    feature_name = config['dataset']['feature_name']

    # decide which edge index to use in this experiment
    graph_params = config['dataset']['graph_params']
    spatial_key = graph_params['spatial_key']
    delaunay = graph_params['delaunay']
    n_neighs = graph_params['n_neighs']
    radius = graph_params['radius']

    assert delaunay or n_neighs or radius, "Specify at least one of delaunay, n_neighs, or radius."

    edge_index_name = set_edge_index_name(
                        spatial_key=spatial_key,
                        delaunay=delaunay,
                        n_neighs=n_neighs,
                        radius=radius,
                    )

    # decide which node labels to use in this experiment
    label_name = config['dataset']['label_name']

    # --------------------- Initialize Data Transforms ---------------------
    ExperimentDataKeys = SetExperimentDataKeys(
                            feature_name=feature_name,
                            label_name=label_name,
                            edge_index_name=edge_index_name
                        )

    # e.g. normalize features , train-val-test split, etc.
    data_transform_names = config['dataset']['transform_names']
    transform_params = config['dataset']['transform_params']
    DataTransforms = init_data_transforms(
                        data_transform_names=data_transform_names,
                        **transform_params
                    )

    # initialize a composed transform
    transforms = T.Compose([ExperimentDataKeys] + DataTransforms)

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
    if model_name == 'MLP':
        Model = MLP
    elif model_name in ['GraphSAGE', 'GATv2', 'GIN']:
        Model = VanillaGNN
    elif model_name == 'VQGraph':
        Model = VQGraph
    else:
        raise ValueError(f"Model {model_name} not found.")
    return Model


def initialize_model(
        config: Dict,
        in_channels: int,
        out_channels: int,
    ) -> pl.LightningModule:
    # set model class
    Model = set_model_class(model_name=config['model']['model_name'])

    # get model, optimizer, loss, and task parameters
    model_name = config['model']['model_name']
    encoder_name = config['model']['encoder_name']
    attribute_decoder_name = config['model']['attribute_decoder_name']
    predictor_name = config['model']['predictor_name']
    train_log_flags = config['model']['train_log_flags']
    encoder_params = config['model']['encoder_params']
    optimizer_params = config['model']['optimizer_params']
    loss_params = config['model']['loss_params']

    # initialize model
    model = Model(
                model_name=model_name,
                encoder_name=encoder_name,
                attribute_decoder_name=attribute_decoder_name,
                predictor_name=predictor_name,
                **train_log_flags,
                in_channels=in_channels,
                out_channels=out_channels,
                **encoder_params,
                **optimizer_params,
                **loss_params,
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