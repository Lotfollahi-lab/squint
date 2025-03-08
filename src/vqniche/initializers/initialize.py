from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader as BatchBuilder
from pytorch_lightning.loggers import WandbLogger

from ..preprocessors.graph_constructors import set_edge_index_name
from ..dataset.transforms import SetExperimentDataKeys, init_data_transforms
from ..dataset.in_memory_dataset_blob import InMemoryDatasetBlob
from ..dataloaders.in_memory_datamodule import InMemoryDataModule
from ..models.graphsage import GraphSAGE
from ..models.vqgraph import VQGraph


def initialize_logger(
        config: Dict,
    ) -> WandbLogger:
    wandb_tags = [
                    config['dataset']['dataset_name'],
                    f"batch={config['dataset']['adata_batch_idx']}",
                    config['model']['model_name'],
                ]

    wandb_log_dir = set_wandb_log_dir(config)
    logger = WandbLogger(
                project="VQNiche",
                save_dir=wandb_log_dir,
                log_model=config['logging']['log_model'],
                offline=config['logging']['offline'],
                tags=wandb_tags,
            )

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


def initialize_model(
        config: Dict,
        in_channels: int,
        out_channels: int,
    ) -> pl.LightningModule:
    # get model, optimizer, loss, and task parameters
    model_name = config['model']['model_name']
    encoder_name = config['model']['encoder_name']
    predictor_name = config['model']['predictor_name']
    encoder_params = config['model']['encoder_params']
    optimizer_params = config['model']['optimizer_params']
    loss_params = config['model']['loss_params']
    task_params = config['model']['task_params']
    inference_params = config['model']['inference_params']

    # initialize model
    if model_name == 'GraphSAGE':
        Model = GraphSAGE
    elif model_name == 'VQGraph':
        Model = VQGraph
    else:
        raise ValueError(f"Model {model_name} not found.")
    model = Model(
                model_name=model_name,
                encoder_name=encoder_name,
                predictor_name=predictor_name,
                in_channels=in_channels,
                out_channels=out_channels,
                **encoder_params,
                **optimizer_params,
                **loss_params,
                **task_params,
                **inference_params,
            )
    return model


def set_wandb_log_dir(
        config: Dict,
    ) -> Path:

    wandb_log_dir = Path(config['logging']['root_log_dir']) / config['dataset']['dataset_name'] / config['experiment']['mode']

    wandb_log_dir = wandb_log_dir / f"batch={config['dataset']['adata_batch_idx']}"

    edge_index_name = set_edge_index_name(
                        spatial_key=config['dataset']['graph_params']['spatial_key'],
                        delaunay=config['dataset']['graph_params']['delaunay'],
                        n_neighs=config['dataset']['graph_params']['n_neighs'],
                        radius=config['dataset']['graph_params']['radius'],
                    )
    wandb_log_dir = wandb_log_dir / f"{edge_index_name}_{config['dataset']['label_name']}"

    wandb_log_dir = wandb_log_dir / f"trainratio={1-config['dataset']['transform_params']['val_ratio']-config['dataset']['transform_params']['test_ratio']}"

    wandb_log_dir = wandb_log_dir / f"{config['datamodule']['loader_name']}_batchsize={config['datamodule']['loader_params']['batch_size']}_neighbors={'_'.join([str(i) for i in config['datamodule']['sampler_params']['num_neighbors']])}"

    wandb_log_dir = wandb_log_dir / config['model']['model_name']

    wandb_log_dir.mkdir(parents=True, exist_ok=True)

    return wandb_log_dir