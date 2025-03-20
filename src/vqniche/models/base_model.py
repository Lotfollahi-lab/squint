import pandas as pd
from pathlib import Path

import torch
import torch_geometric
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Accuracy
from typing import List, Tuple, Callable, Optional, Literal

from ..utils.loss import *


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str = 'BaseModel',
            encoder_name: str = 'GraphSAGE',
            predictor_name: str = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'mean'},
            task_name: str = 'multiclass',
            task_kwargs: dict = {},
            inference_mode: Literal['batch-wise', 'layer-wise'] = 'batch-wise',
        ) -> None:
        """
        Initialize the BaseModel class.

        Parameters
        ----------
        - model_name: str
            The name of the model.
        - encoder_name: str
            The encoder name.
        - predictor_name: str
            The predictor name.

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

        - task_name: str
            The task name.
        - task_kwargs: dict
            The task keyword arguments.

        - inference_mode: Literal['batch-wise', 'layer-wise']
            The inference mode. Choose from 'batch-wise' or 'layer-wise'.
        """
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.predictor_name = predictor_name

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
        self.loss_fn_tuples = self.set_loss_fn_tuples(loss_names, loss_kwargs)

        # Accuracy metrics parameters
        self.task_name = task_name
        self.task_kwargs = task_kwargs
        self.train_acc = Accuracy(
                            task=task_name,
                            num_classes=out_channels,
                            **task_kwargs
                            )
        self.val_acc = Accuracy(
                            task=task_name,
                            num_classes=out_channels,
                            **task_kwargs
                            )
        self.test_acc = Accuracy(
                            task=task_name,
                            num_classes=out_channels,
                            **task_kwargs
                            )

        # Option 1 -- batch-wise (default)
        # for epoch in epochs:
        #    for batch in loader:
        #       for layer in model.layers:
        #         layer.forward(batch)
        # Option 2 -- layer-wise
        # for epoch in epochs:
        #    for layer in model.layers:
        #       for batch in loader:
        #         layer.forward(batch)
        self.inference_mode = inference_mode

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

                loss_fn_data_keys = ['pred_attr', 'target_attr']

                distribution = loss_kwargs.get('distribution')
                if 'distribution' is not None:
                    loss_fn_params['distribution'] = distribution

                dispersion_theta = loss_kwargs.get('dispersion_theta')
                if dispersion_theta is not None:
                    loss_fn_params['dispersion_theta'] = dispersion_theta

                wt_attr_reconstr = loss_kwargs.get('wt_attr_reconstr')
                if wt_attr_reconstr is not None:
                    loss_fn_params['wt_attr_reconstr'] = wt_attr_reconstr

            elif loss_fn_name == 'mse_adjacency_reconstruction':
                loss_fn = mse_adjacency_reconstruction

                loss_fn_data_keys = ['pred_adj', 'batch_edge_index', 'batch_input_id', 'batch_nid']

                wt_adj_reconstr = loss_kwargs.get('wt_adj_reconstr')
                if wt_adj_reconstr is not None:
                    loss_fn_params['wt_adj_reconstr'] = wt_adj_reconstr

            elif loss_fn_name == 'mse_commitment_loss':
                loss_fn = mse_commitment_loss

                loss_fn_data_keys = ['pred_commit', 'target_commit']

                wt_commit = loss_kwargs.get('wt_commit')
                if wt_commit is not None:
                    loss_fn_params['wt_commit'] = wt_commit

            elif loss_fn_name == 'l2_codebook_loss':
                loss_fn = l2_codebook_loss

                loss_fn_data_keys = ['codebook_embeddings']

                wt_codebook = loss_kwargs.get('wt_codebook')
                if wt_codebook is not None:
                    loss_fn_params['wt_codebook'] = wt_codebook

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


    def log_metrics(
            self,
            mode: str = 'train',
            loss_value: torch.Tensor = None,
            acc_value: torch.Tensor = None,
            curr_batch_size: int = None,
        ) -> None:
        """
        Log total loss (if available) and accuracy for the model during training, validation, and testing.

        Parameters
        ----------
        - mode: str
            The mode of the model (train, val, test).
        - loss_value: torch.Tensor
            The computed loss.
        - acc_value: torch.Tensor
            The computed accuracy.
        - curr_batch_size: int
            The number of samples in the current batch.
        """
        assert acc_value is not None, 'Accuracy value is None'
        if isinstance(acc_value, torchmetrics.Metric):
            raise ValueError(f"Accuracy value is a torchmetrics.Metric object: {acc_value}.")

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


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of a standard GNN model. This is a composition of the forward pass of the encoder and the predictor. The batch of nodes may be the entire set of nodes in the graph or a subset of nodes.

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
        # calls the forward method of the Model's encoder
        batch_node_embeddings = self.encoder(batch_x, batch_edge_index)

        # calls the forward method of the Model's predictor
        unnormalized_logits = self.predictor(batch_node_embeddings)

        return unnormalized_logits


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


    def on_train_epoch_end(self) -> None:
        """
        Callback function to be executed at the end of each training epoch.

        Notes
        -----
        - We use this hook to log the train and validation loss terms and accuracies in the training loop.
        """
        # log the metrics at the end of each epoch
        metric_names = ['epoch'] + list(self.trainer.callback_metrics.keys())
        metrics_values = [self.current_epoch] + [value.item() for value in self.trainer.callback_metrics.values()]
        for metric_name, metric_value in zip(metric_names, metrics_values):
            print(f"{metric_name}: {metric_value}")

        self.train_val_epoch_metrics = pd.concat(
            [self.train_val_epoch_metrics, pd.DataFrame([metrics_values], columns=metric_names)],
            ignore_index=True
        )

        return super().on_train_epoch_end()


    def on_validation_epoch_start(self) -> None:
        """
        Callback function to be executed at the start of each validation epoch.

        Notes:
        ------
        - We use this hook to obtain the unnormalized logits for the validation set if the inference mode is 'layer-wise'. Otherwise, we do nothing.
        """
        if self.inference_mode == 'batch-wise':
            pass

        elif self.inference_mode == 'layer-wise':
            self.val_logits = self.inference(
                                self.trainer.datamodule.val_dataloader()
                                )

        return super().on_validation_epoch_start()


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


    def on_validation_epoch_end(self) -> None:
        """
        Callback function to be executed at the end of each validation epoch.

        Notes:
        ------
        - We use this hook to print the total validation loss at the end of each epoch to monitor the validation progress.
        """
        super().on_validation_epoch_end()


    def on_test_epoch_start(self) -> None:
        """
        Callback function to be executed at the start of each test epoch.

        Notes:
        ------
        - We use this hook to obtain the unnormalized logits for the test set if the inference mode is 'layer-wise'. Otherwise, we do nothing.
        """
        if self.inference_mode == 'batch-wise':
            pass

        elif self.inference_mode == 'layer-wise':
            self.test_logits = self.inference(
                                self.trainer.datamodule.test_dataloader()
                                )

        return super().on_test_epoch_start()


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

    def on_fit_end(self):
        epoch_metrics_fname = Path(self.logger.experiment.dir) / 'train_val_epoch_metrics.csv'
        self.train_val_epoch_metrics.to_csv(
            path_or_buf=epoch_metrics_fname,
            sep=',',
            index=False
        )

        return super().on_fit_end()
