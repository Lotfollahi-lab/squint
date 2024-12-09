import copy
import torch
import torch_geometric
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import List, Tuple, Callable, Optional
import torch.functional as F

from ..utils.loss import cross_entropy_loss


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            name: str = 'BaseModel',
            in_channels: int = None,
            out_channels: int = None,
            encoder_name: str = 'GraphSAGE',
            predictor_name: str = 'Linear',
            hidden_channels: int = 256,
            num_layers: int = 2,
            dropout: float = 0.5,
            lr: float = 0.01,
            weight_decay: float = 0.0,
            optimizer_name: str = 'adam',
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'mean'},
            task_name: str = 'multiclass',
            task_kwargs: dict = {},
            inference_mode: Optional[str] = 'layer-wise',
            **kwargs
        ) -> None:
        """
        Initialize the BaseModel class.

        Parameters
        ----------
        - name: str
            The name of the model.
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - encoder_name: str
            The encoder name.
        - predictor_name: str
            The predictor name.
        - hidden_channels: int
            The number of hidden channels.
        - num_layers: int
            The number of layers.
        - dropout: float
            The dropout rate.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.
        - optimizer_name: str
            The optimizer name.
        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            The loss function keyword arguments.
        - task_name: str
            The task name.
        - task_kwargs: dict
            The task keyword arguments.
        - inference_mode: Optional[str]
            The inference mode. Choose from 'batch-wise' or 'layer-wise'.
        - kwargs: dict
            Additional keyword arguments.
        """
        self.name = name

        super(BaseModel, self).__init__(**kwargs)

        # Encoder parameters
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Predictor parameters
        self.predictor_name = predictor_name
        self.out_channels = out_channels

        # Optimizer parameters
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        # Loss parameters
        self.loss_kwargs = loss_kwargs
        self.loss_fn_tuples = self.set_loss_fn_tuples(loss_names, loss_kwargs)

        # Accuracy metrics parameters
        self.task_name = task_name
        self.task_kwargs = task_kwargs
        self.train_acc = Accuracy(task=task_name, num_classes=out_channels, **task_kwargs)
        self.val_acc = Accuracy(task=task_name, num_classes=out_channels, **task_kwargs)
        self.test_acc = Accuracy(task=task_name, num_classes=out_channels, **task_kwargs)

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
        if inference_mode not in ['batch-wise', 'layer-wise']:
            raise ValueError(f"Invalid inference mode: {inference_mode}.")
        self.inference_mode = inference_mode

        self.save_hyperparameters()


    def set_loss_fn_tuples(
            self,
            loss_fn_names: List[str],
            loss_kwargs: dict = {}
        ) -> List[Tuple[str, Callable, List[str], dict]]:
        """
        Set the loss functions for the encoder.

        Parameters
        ----------
        - loss_names: List[str]
            The loss function names.
        - loss_kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        List
            One tuple per loss name in loss_names comprising of loss function name (str), loss function (callable), list of data related key strings required to be passed to the loss function, and a dictionary of additional keyword arguments for the loss function.

        Notes
        -----
        Use this method to set the loss functions for the encoder. The loss functions should be defined in the utils.loss module.
        """
        # initialize a list to store the loss function tuples
        loss_fn_tuples = []

        for loss_fn_name in loss_fn_names:
            loss_fn_params = {}
            if loss_fn_name == 'cross_entropy':
                # set the cross-entropy loss function
                loss_fn = cross_entropy_loss

                # set key names for data required to compute cross-entropy loss
                loss_fn_data_keys = ['logits', 'labels']

                # set keyword parameters for cross-entropy loss
                reduction = loss_kwargs.get('reduction')
                if reduction is not None:
                    loss_fn_params['reduction'] = reduction
            else:
                raise NotImplementedError(f'{loss_fn_name} Loss not implemented')

            loss_fn_tuple = (loss_fn_name, loss_fn, loss_fn_data_keys, loss_fn_params)
            loss_fn_tuples.append(loss_fn_tuple)

        return loss_fn_tuples


    def criterion(
            self,
            loss_data: dict,
            curr_batch_size: Optional[int] = None
        ) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters
        ----------
        - loss_data: dict
            A collection of data objects required to compute the loss.
        - curr_batch_size: int
            The number of samples in the current batch. Required for logging.

        Returns
        -------
        - torch.Tensor
            The computed loss.
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
            total_loss += loss_fn_value

            # log each computed loss term
            self.log(
                    name=loss_fn_name,
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
        - torch.Tensor
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
            return torch.optim.Adam(params=self.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
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


    def on_validation_epoch_start(self) -> None:
        if self.inference_mode == 'batch-wise':
            print("Applying inference Batch-wise for Validation")

        elif self.inference_mode == 'layer-wise':
            print("Applying inference Layer-wise for Validation")
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


    def on_test_epoch_start(self) -> None:
        if self.inference_mode == 'batch-wise':
            print("Applying inference Batch-wise for Testing")

        elif self.inference_mode == 'layer-wise':
            print("Applying inference Layer-wise for Testing")
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