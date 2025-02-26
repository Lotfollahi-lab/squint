from pathlib import Path

import torch
import pytorch_lightning as pl
import torch_geometric.transforms as T
from pytorch_lightning.loggers import WandbLogger

from ..preprocessors.graph_constructors import set_edge_index_name
from ..dataset.transforms import SetExperimentDataKeys, init_data_transforms
from ..dataset.in_memory_dataset_blob import InMemoryDatasetBlob
from ..dataloaders.in_memory_datamodule import InMemoryDataModule
from ..models.graphsage import GraphSAGE
from ..models.vqgraph import VQGraph


def initialize_logger(
        config: dict,
        param_strings: dict,
    ):
    dataset_name = config['dataset']['dataset_name']
    model_name = config['model']['model_name']

    # --------------------- Wandb ---------------------
    if config['logging']['enabled'] :
        # set logging directory
        wandb_log_dir = Path(config['logging']['root_log_dir']) / dataset_name / config['experiment']['mode']
        wandb_log_dir = wandb_log_dir / param_strings['batch'] / param_strings['datakeys'] / param_strings['transform'] / param_strings['loader_sampler'] / model_name
        wandb_log_dir.mkdir(parents=True, exist_ok=True)

        wandb_tags = [
                        dataset_name,
                        model_name,
                        param_strings['batch'],
                    ]

        # initialize wandb logger
        logger = WandbLogger(
                        project="VQNiche",
                        save_dir=wandb_log_dir,
                        log_model=config['logging']['log_model'],
                        offline=config['logging']['offline'],
                        tags=wandb_tags,
                    )

        # configure model checkpointing
        ckpt_log_dir = Path(logger.experiment.dir) / 'checkpoints'

    else:
        logger = None
        ckpt_log_dir = Path("./test/checkpoints")

    ckpt_log_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Checkpoints ---------------------
    checkpoint_params = config['trainer']['checkpoint_params']
    checkpoints = [
                    pl.callbacks.ModelCheckpoint(
                        dirpath=ckpt_log_dir,
                        filename='{epoch}-{val_acc:.2f}',
                        **checkpoint_params
                        )
                    ]

    return logger, \
            checkpoints



def initialize_data_and_model(config: dict):
    param_strings = {}

    # --------------------- Determinism Settings ---------------------
    # set seed for reproducibility
    seed = config['experiment']['seed']
    pl.seed_everything(seed)

    # set backend deterministic as true
    torch.backends.cudnn.deterministic = True

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

    # set data keys directory name
    datakeys_param_str = f"{edge_index_name}_{label_name}"
    param_strings['datakeys'] = datakeys_param_str

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

    # set transform directory name
    transform_param_str = f"trainratio={1-transform_params['val_ratio']-transform_params['test_ratio']}"
    param_strings['transform'] = transform_param_str

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
    batch_idx = config['dataset']['batch_idx']

    # set batch directory name
    batch_idx_param_str = f"batch={batch_idx}"
    param_strings['batch'] = batch_idx_param_str

    # --------------------- Load Data (one batch) ---------------------
    # load PyG data object corresponding to batch_idx (e.g. AnnData batch0)
    data_batch = dataset_blob[batch_idx]
    assert batch_idx == data_batch.batch, f"Batch index mismatch: {batch_idx} != {data_batch.batch}"

    # --------------------- Initialize DataModule ---------------------
    # set parameters for data loader and sampler for training, validation, and testing
    loader_name = config['datamodule']['loader_name']
    loader_params = config['datamodule']['loader_params']

    sampler_name = config['datamodule']['sampler_name']
    sampler_params = config['datamodule']['sampler_params']

    inference_params = config['datamodule']['inference_params']

    datamodule_batch = InMemoryDataModule(
                            data=data_batch,
                            loader_name=loader_name,
                            loader_params=loader_params,
                            sampler_name=sampler_name,
                            sampler_params=sampler_params,
                            **inference_params,
                        )
    # set loader and sampler directory name
    loader_sampler_param_str = f"{loader_name}_batchsize={loader_params['batch_size']}_neighbors={'_'.join([str(i) for i in sampler_params['num_neighbors']])}"
    param_strings['loader_sampler'] = loader_sampler_param_str

    # --------------------- Initialize Model ---------------------
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
                in_channels=data_batch.num_features,
                out_channels=data_batch.num_classes,
                **encoder_params,
                **optimizer_params,
                **loss_params,
                **task_params,
                **inference_params,
            )

    return data_batch, \
            datamodule_batch, \
            model, \
            param_strings

