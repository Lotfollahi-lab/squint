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
        - kwargs: dict
            Additional keyword arguments.
        """
        self.name = name

        self.save_hyperparameters()

        super().__init__(**kwargs)

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
            batch_size: Optional[int] = None
        ) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters
        ----------
        - loss_data: dict
            A collection of data objects required to compute the loss.
        - batch_size: int
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
                    batch_size=batch_size,
                    sync_dist=True,
                    )

            # free up memory by deleting the intermediate loss function data
            del _loss_fn_data

        return total_loss


    def common_step(
            self,
            data: torch_geometric.data.Data
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common step across training, validation, and testing for the model.

        Parameters
        ----------
        - data: torch_geometric.data.Data
            The input data.

        Returns
        -------
        - torch.Tensor
            The unnormalized logits.
        - torch.Tensor
            The predicted class probabilities.
        - torch.Tensor
            The ground truth labels.
        """
        # TODO: Update data to batch if applicable
        unnormalized_logits = self(data.x, data.edge_index)[:data.batch_size]
        preds = unnormalized_logits.softmax(dim=-1) # predicted class probabilities
        labels = data.y[:data.batch_size]
        return unnormalized_logits, preds, labels


    def training_step(
            self,
            train_loss: torch.tensor,
            preds: torch.Tensor,
            labels: torch.Tensor,
            train_batch_size: Optional[int] = None
        ) -> None:
        """
        Training step for the model.

        Parameters
        ----------
        - train_loss: torch.tensor
            The computed loss.
        - preds: torch.Tensor
            The predicted class probabilities.
        - labels: torch.Tensor
            The ground truth labels.
        - train_batch_size: int
            The number of samples in the current training batch.

        Returns
        -------
        - None
            Child classes must return training loss.

        Notes
        -----
        Throws an error if the loss is not computed by the child class. The default implementation logs the training loss, computes the training accuracy, and logs the training accuracy.
        """
        assert train_loss is not None, 'Train Loss not computed'

        self.log(
                name='train_loss',
                value=train_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=train_batch_size,
                sync_dist=True,
                )

        self.train_acc(preds, labels)
        self.log(
                name='train_acc',
                value=self.train_acc,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=train_batch_size,
                sync_dist=True,
                )


    def validation_step(
            self,
            val_loss: torch.tensor,
            preds: torch.Tensor,
            labels: torch.Tensor,
            val_batch_size: Optional[int] = None
        ) -> None:
        """
        Validation step for the model.

        Parameters
        ----------
        - val_loss: torch.tensor
            The computed loss.
        - preds: torch.Tensor
            The predicted class probabilities.
        - labels: torch.Tensor
            The ground truth labels.
        - val_batch_size: int
            The number of samples in the current validation batch.

        Returns
        -------
        - None
            Child classes must return validation loss.

        Notes
        -----
        Throws an error if the loss is not computed by the child class. The default implementation logs the validation loss, computes the validation accuracy, and logs the validation accuracy.
        """
        assert val_loss is not None, 'Validation Loss not computed'

        self.log(
                name='val_loss',
                value=val_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=val_batch_size,
                sync_dist=True,
                )

        self.val_acc(preds, labels)
        self.log(
                name='val_acc',
                value=self.val_acc,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=val_batch_size,
                sync_dist=True,
                )


    def test_step(
            self,
            test_data: torch_geometric.data.Data
        ) -> None:
        """
        Test step for the model.

        Parameters
        ----------
        - data: torch_geometric.data.Data
            The input data.

        Notes
        -----
        This method may be overridden by subclasses to define the test step. The default implementation computes the test accuracy and logs it.
        """
        test_batch_size = test_data.size(0)

        _, preds, labels = self.common_step(test_data)

        self.test_acc(preds, labels)
        self.log(
                name='test_acc',
                value=self.test_acc,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=test_batch_size,
                sync_dist=True,
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
            return torch.optim.Adam(params=self.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {self.optimizer_name} not implemented')


    def forward(self):
        """
        Forward pass of the model.

        Notes
        -----
        This method should be overridden by subclasses to define the forward pass.
        """
        raise NotImplementedError('Forward method not implemented')