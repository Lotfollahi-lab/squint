from pathlib import Path
from typing import List, Tuple, Callable, Optional, Literal

import pandas as pd

import torch
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.nn.dense.linear import Linear

from vqniche import metrics
from ..utils.loss import *
from ..modules.mlp import MLP as MLP_AdjacencyDecoder
from ..decoders.mlp_softmax import MLPSoftmax


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            encoder_name: str,
            attribute_decoder_name: str,
            adjacency_decoder_name: str,
            predictor_name: str,
            log_similarity_stats: bool = False,
            log_pearson_correlation: bool = False,
            log_mmd_degree: bool = False,
            log_codebook_utilization: Optional[bool] = None,
            in_channels: int = None,
            out_channels: int = None,
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'mean'},
        ) -> None:
        """
        Initialize the BaseModel class.

        Parameters
        ----------
        - model_name: str
            The name of the model.
        - encoder_name: str
            The encoder name.
        - attribute_decoder_name: str
            The name of the attribute decoder module.
        - adjacency_decoder_name: str
            The name of the adjacency decoder module.
        - predictor_name: str
            The name of the predictor module.
        - log_similarity_stats: bool
            Whether to log the similarity statistics.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation.
        - log_mmd_degree: bool
            Whether to log the MMD metric between the degree distribution of the original and reconstructed graphs.
        - log_codebook_utilization: bool
            Whether to log the codebook utilization.

        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.

        - optimizer_name: str
            The optimizer name.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.

        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            The loss function keyword arguments.

        """
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.attribute_decoder_name = attribute_decoder_name
        self.adjacency_decoder_name = adjacency_decoder_name
        self.predictor_name = predictor_name
        self.log_similarity_stats = log_similarity_stats
        self.log_pearson_correlation = log_pearson_correlation
        self.log_mmd_degree = log_mmd_degree
        if log_codebook_utilization is not None:
            self.log_codebook_utilization = log_codebook_utilization

        super().__init__()

        # Data parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Optimizer parameters
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay

        # Loss parameters
        self.loss_kwargs = loss_kwargs
        self.dispersion = torch.nn.Parameter(torch.randn(self.in_channels))
        self.loss_fn_tuples = self.set_loss_fn_tuples(loss_names, loss_kwargs)

        # Inference mode: Batch-wise
        # for epoch in epochs:
        #    for batch in loader:
        #       for layer in model.layers:
        #         layer.forward(batch)

        self.train_val_epoch_metrics = pd.DataFrame()

        self.save_hyperparameters()


    def set_loss_fn_tuples(
            self,
            loss_fn_names: List[str],
            loss_kwargs: dict = {}
        ) -> List[Tuple[str, Callable, List[str], dict]]:
        """
        Set the loss functions for the model.

        Parameters
        ----------
        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        loss_fn_tuples: List
            One tuple per loss name in loss_names comprising of loss function name (str), loss function (callable), list of data related key strings required to be passed to the loss function, and a dictionary of additional keyword arguments for the loss function.

        Notes
        -----
        Use this method to set the loss functions for the encoder. The loss functions should be defined in the utils.loss module.
        """
        # initialize a list to store the loss function tuples
        loss_fn_tuples = []

        print("Setting the following loss terms as criterion for training:")
        for loss_fn_name in loss_fn_names:
            print(f"Loss function: {loss_fn_name}")
            loss_fn_params = {}

            if loss_fn_name == 'cross_entropy':
                # set the cross-entropy loss function
                loss_fn = cross_entropy

                # set key names for data required to compute cross-entropy loss
                loss_fn_data_keys = ['logits', 'labels']

                # set keyword parameters for cross-entropy loss
                wt_cross_entropy = loss_kwargs.get('wt_cross_entropy')
                if wt_cross_entropy is not None:
                    loss_fn_params['wt_cross_entropy'] = wt_cross_entropy

            elif loss_fn_name == 'mse_attribute_reconstruction':
                loss_fn = mse_attribute_reconstruction

                loss_fn_data_keys = ['pred_attr', 'target_attr']

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'nb_attribute_reconstruction':
                loss_fn = nb_attribute_reconstruction

                loss_fn_data_keys = ['pred_attr', 'target_attr', 'edge_index', 'batch_size', 'dispersion']

                k_hop_nb_loss = loss_kwargs.get('k_hop_nb_loss')
                if k_hop_nb_loss is not None:
                    loss_fn_params['k_hop_nb_loss'] = k_hop_nb_loss

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'mse_adjacency_reconstruction':
                loss_fn = mse_adjacency_reconstruction

                loss_fn_data_keys = ['batch_size', 'h_adj', 'batch_edge_index']

                estimate_adj_kwargs = loss_kwargs.get('estimate_adj_kwargs')
                if estimate_adj_kwargs is not None:
                    loss_fn_params['estimate_adj_kwargs'] = estimate_adj_kwargs

                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'mse_joint_code_commit_loss':
                loss_fn = mse_joint_code_commit_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_joint_code_commit = loss_kwargs.get('wt_joint_code_commit')
                if wt_joint_code_commit is not None:
                    loss_fn_params['wt_joint_code_commit'] = wt_joint_code_commit

            elif loss_fn_name == 'mse_commit_loss':
                loss_fn = mse_commit_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_commit = loss_kwargs.get('wt_commit')
                if wt_commit is not None:
                    loss_fn_params['wt_commit'] = wt_commit

            elif loss_fn_name == 'mse_code_loss':
                loss_fn = mse_code_loss

                loss_fn_data_keys = ['quantizer_input', 'quantizer_output']

                wt_code = loss_kwargs.get('wt_code')
                if wt_code is not None:
                    loss_fn_params['wt_code'] = wt_code

            elif loss_fn_name == 'l2_codebook_orthogonal_regularization_loss':
                loss_fn = l2_codebook_orthogonal_regularization_loss

                loss_fn_data_keys = ['codebook_embeddings']

                wt_codebook_orthogonal_regularization = loss_kwargs.get('wt_codebook_orthogonal_regularization')
                if wt_codebook_orthogonal_regularization is not None:
                    loss_fn_params['wt_codebook_orthogonal_regularization'] = wt_codebook_orthogonal_regularization

                codebook_reg_active_codes_only = loss_kwargs.get('codebook_reg_active_codes_only')
                if codebook_reg_active_codes_only is not None:
                    loss_fn_params['codebook_reg_active_codes_only'] = codebook_reg_active_codes_only

                codebook_reg_max_codes = loss_kwargs.get('codebook_reg_max_codes')
                if codebook_reg_max_codes is not None:
                    loss_fn_params['codebook_reg_max_codes'] = codebook_reg_max_codes

            else:
                raise NotImplementedError(f'{loss_fn_name} Loss not implemented')

            loss_fn_tuple = (loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params)
            loss_fn_tuples.append(loss_fn_tuple)

        return loss_fn_tuples


    def criterion(
            self,
            loss_data: dict,
            curr_batch_size: Optional[int] = None,
            mode: Literal['train', 'val'] = 'train',
        ) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters
        ----------
        - loss_data: dict
            A collection of data objects required to compute the loss.
        - curr_batch_size: int
            The number of samples in the current batch. Required for logging.
        - mode: Literal['train', 'val']
            The mode of the model (train, val).

        Returns
        -------
        - total_loss: torch.Tensor
            The total computed loss across all loss terms for the current batch.
        """
        assert len(self.loss_fn_tuples) > 0, 'No loss functions defined'

        # initialize total_loss = 0.0 with requires_grad=True so that the loss can be backpropagated
        # total_loss will be computed as the sum of all the loss terms from self.loss_names
        total_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float32).to(self.device)

        # during model initialization, self.loss_fn_tuples is set to a list of tuples
        # one tuple per loss name in self.loss_names
        # each tuple contains the loss function name, the callable loss function, the data keys required to compute the loss, and the loss function parameters
        for loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params in self.loss_fn_tuples:
            # extract from loss_data, the data required to compute the current loss function
            _loss_fn_data = {key: loss_data[key] for key in loss_fn_data_keys}

            # pass the extracted data to the loss function along with the loss function related kwargs
            loss_fn_value = loss_fn(**_loss_fn_data, **loss_fn_params)
            # add the computed loss to the total_loss
            total_loss = torch.add(total_loss, loss_fn_value)

            # log each computed loss term
            self.log(
                    name=f"{mode}_{loss_fn_name}",
                    value=loss_fn_value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=curr_batch_size,
                    sync_dist=True,
                    )

            # free up memory by deleting the intermediate loss function data
            del _loss_fn_data

        return total_loss


    def _init_attribute_decoder(
            self,
            in_channels: int = None,
            out_channels: int = None,
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            attribute_decoder_params: dict = {}
        ) -> pl.LightningModule:
        """
        Initialize the attribute decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the attribute decoder module.
        - out_channels: int
            The output dimension of the attribute decoder module.
        - attribute_decoder_name: Literal['MLPSoftmax']
            The name of the attribute decoder module.
        - attribute_decoder_params: dict
            The parameters for the attribute decoder module.

        Returns
        -------
        - attribute_decoder: pl.LightningModule
            The attribute decoder module.
        """
        if attribute_decoder_name == 'MLPSoftmax':
            return MLPSoftmax(
                in_channels=in_channels,
                out_channels=out_channels,
                name=attribute_decoder_name,
                use_xy_coordinates=attribute_decoder_params['use_xy_coordinates'],
                mlp_params=attribute_decoder_params['mlp_params'],
            )


    def _init_adjacency_decoder(
            self,
            in_channels: int,
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            adjacency_decoder_params: dict = {}
        ) -> torch.nn.Module:
        """
        Initialize the adjacency decoder module.

        Parameters
        ----------
        - in_channels: int
            The input dimension of the adjacency decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - adjacency_decoder_params: dict
            The parameters for the adjacency decoder module.

        Returns
        -------
        - adjacency_decoder: torch.nn.Module
            The adjacency decoder module.
        """
        if adjacency_decoder_name == 'MLP_AdjacencyDecoder':
            if 'out_channels' not in adjacency_decoder_params:
                adjacency_decoder_params['out_channels'] = in_channels
            return MLP_AdjacencyDecoder(
                in_channels=in_channels,
                out_channels=adjacency_decoder_params['out_channels'],
                mlp_params=adjacency_decoder_params['mlp_params'],
            )


    def _init_predictor(
            self,
            predictor_name: Literal['Linear'] = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            init_method: str = 'kaiming_uniform'
        ) -> pl.LightningModule:
        """
        Initialize the predictor module.

        Parameters
        ----------
        - predictor_name: str
            The name of the predictor module.
        - in_channels: int
            The input dimension of the predictor module.
        - out_channels: int
            The output dimension of the predictor module.
        - init_method: str
            The initialization method for the predictor module.

        Returns
        -------
        - predictor: pl.LightningModule
            The predictor module.
        """
        if predictor_name == 'Linear':
            return Linear(
                in_channels=in_channels,
                out_channels=out_channels,
                weight_initializer=init_method
            )


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns
        -------
        - torch.optim.Optimizer
            The configured optimizer.
        """
        # TODO: Add support for multiple optimizers
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f'Optimizer {self.optimizer_name} not implemented')


    def common_step(
            self,
            batch_loss_data: dict,
            batch_size: int,
            mode: Literal['train', 'val', 'test'] = 'train',
        ) -> torch.Tensor:
        """
        Compute the loss for a model for a given batch if mode is 'train' or 'val', but not if mode is 'test'.
        Log the loss (if available) and accuracy for the current batch.

        Parameters
        ----------
        - batch_loss_data: dict
            The data required to compute the loss.
        - batch_size: int
            The size of the batch.
        - mode: Literal['train', 'val', 'test']
            The mode of the model (train, val, test).

        Returns
        -------
        - torch.Tensor
            The computed loss for the current batch if mode is 'train' or 'val'. The computed accuracy for the current batch if mode is 'test'.
        """
        if mode in ['train', 'val']:
            loss_value = self.criterion(
                loss_data=batch_loss_data,
                curr_batch_size=batch_size,
            )
        elif mode == 'test':
            loss_value = None

        acc_value = metrics.accuracy_score(
            unnormalized_logits=batch_loss_data['logits'],
            one_hot_labels=batch_loss_data['labels'],
        )

        self.log_metrics(
            loss_value=loss_value,
            acc_value=acc_value,
            curr_batch_size=batch_size,
            mode=mode,
        )

        if mode in ['train', 'val']:
            return loss_value
        elif mode == 'test':
            return acc_value


    def log_metrics(
            self,
            loss_value: torch.Tensor = None,
            acc_value: torch.Tensor = None,
            curr_batch_size: int = None,
            mode: Literal['train', 'val', 'test'] = 'train',
        ) -> None:
        """
        Log total loss (if available) and accuracy for the model during training, validation, and testing.

        Parameters
        ----------
        - loss_value: torch.Tensor
            The computed loss.
        - acc_value: torch.Tensor
            The computed accuracy.
        - curr_batch_size: int
            The number of samples in the current batch.
        - mode: Literal['train', 'val', 'test']
            The mode of the model (train, val, test).
        """
        assert acc_value is not None, 'Accuracy value is None'

        if loss_value is not None:
            self.log(
                name=f'{mode}_loss',
                value=loss_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=curr_batch_size,
                sync_dist=True,
            )

        self.log(
            name=f'{mode}_acc',
            value=acc_value,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=curr_batch_size,
            sync_dist=True,
        )


    def on_train_epoch_end(self) -> None:
        """
        Callback function to be executed at the end of each training epoch.

        Notes
        -----
        - We use this hook to log the train and validation loss terms and accuracies in the training loop.
        """
        # compute the training epoch end stats such as embedding similarity, pearson correlation, codebook utilization, etc.
        train_epoch_end_stats = self.compute_train_epoch_stats()
        for key, value in train_epoch_end_stats.items():
            self.log(
                name=key,
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # log the metrics at the end of each epoch
        metric_names = ['epoch'] + list(self.trainer.callback_metrics.keys())
        metrics_values = [self.current_epoch] + [value.item() for value in self.trainer.callback_metrics.values()]
        print("--------------------------------")
        for metric_name, metric_value in zip(metric_names, metrics_values):
            print(f"{metric_name}: {metric_value}")
        print("--------------------------------\n")

        self.train_val_epoch_metrics = pd.concat(
            [self.train_val_epoch_metrics, pd.DataFrame([metrics_values], columns=metric_names)],
            ignore_index=True
        )

        return super().on_train_epoch_end()


    def on_fit_end(self):
        epoch_metrics_fname = Path(self.logger.experiment.dir) / 'train_val_epoch_metrics.csv'
        self.train_val_epoch_metrics.to_csv(
            path_or_buf=epoch_metrics_fname,
            sep=',',
            index=False
        )

        return super().on_fit_end()


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the model. The batch of nodes may be the entire set of nodes in the graph or a subset of nodes.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.

        Returns
        -------
        - unnormalized_logits: torch.Tensor
            The unnormalized logits of the model.
        """
        raise NotImplementedError('Forward pass should be implemented by the subclass')


    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Training step for the model.

        Parameters
        ----------
        - train_batch: torch_geometric.data.Data
            The training batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed loss.
        """
        raise NotImplementedError('Training step not implemented')


    def compute_train_epoch_stats(self) -> dict:
        """
        Compute the training epoch end stats such as embedding similarity, pearson correlation, codebook utilization, etc.

        Returns
        -------
        - dict
            The training epoch end stats.
        """
        raise NotImplementedError('Training epoch end stats not implemented')


    def validation_step(
            self,
            val_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Validation step for the model.

        Parameters
        ----------
        - val_batch: torch_geometric.data.Data
            The validation batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed validation loss.
        """
        raise NotImplementedError('Validation step not implemented')


    def test_step(
            self,
            test_batch: torch_geometric.data.Data,
            batch_idx: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Test step for the model.

        Parameters
        ----------
        - test_batch: torch_geometric.data.Data
            The test batch.
        - batch_idx: int
            The batch index.

        Returns
        -------
        - torch.Tensor
            The computed test accuracy.
        """
        raise NotImplementedError('Test step not implemented')