import torch
import torch_geometric
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import List, Tuple

from ..utils.loss import cross_entropy_loss


class BaseModel(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 optimizer_name: str = 'adam',
                 loss_names: List[str] = ['cross_entropy'],
                 task: str = 'multiclass',
                 **kwargs):
        super().__init__()

        # Encoder parameters
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Predictor parameters
        self.out_channels = out_channels

        # Training parameters
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        # Set loss functions
        self.loss_names = loss_names
        self.loss_fn_tuples = self.set_loss_fn_tuples(kwargs)

        # Initialize the accuracy metrics based on the task
        self.task = task
        self.train_acc = Accuracy(task=task, num_classes=out_channels)
        self.val_acc = Accuracy(task=task, num_classes=out_channels)
        self.test_acc = Accuracy(task=task, num_classes=out_channels)


    def set_loss_fn_tuples(self,
                           loss_kwargs: dict) -> List:
        """
        Set the loss functions for the encoder.

        Parameters
        ----------
        loss_kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        List
            The loss functions.

        Notes
        -----
        Use this method to set the loss functions for the encoder. The loss functions should be defined in the utils.loss module.
        """
        # initialize a list to store the loss function tuples
        loss_fn_tuples = []

        for loss_name in self.loss_names:
            loss_fn_params = {}
            if loss_name == 'cross_entropy':
                # set the cross-entropy loss function
                loss_fn = cross_entropy_loss

                # set key names for data required to compute cross-entropy loss
                loss_fn_data_keys = ['logits', 'labels']

                # set keyword parameters for cross-entropy loss
                reduction = loss_kwargs.get('reduction')
                if reduction is not None:
                    loss_fn_params['reduction'] = reduction
            else:
                raise NotImplementedError(f'{loss_name} Loss not implemented')

            loss_fn_tuple = (loss_fn, loss_fn_data_keys, loss_fn_params)
            loss_fn_tuples.append(loss_fn_tuple)

        return loss_fn_tuples


    def criterion(self,
                loss_data: dict) -> torch.Tensor:
        """
        Compute the loss for the model.

        Parameters
        ----------
        loss_data : dict
            A collection of data objects required to compute the loss.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        assert len(self.loss_fn_tuples) > 0, 'No loss functions defined'

        loss = torch.Tensor(0.0)
        for loss_fn, loss_fn_data_keys, loss_fn_params in self.loss_fn_tuples:
            _loss_fn_data = {key: loss_data[key] for key in loss_fn_data_keys}
            loss += loss_fn(**_loss_fn_data, **loss_fn_params)
            del _loss_fn_data
        return loss


    def common_step(self,
                    data: torch_geometric.data.Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common step across training, validation, and testing for the model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch.Tensor
            The unnormalized logits.
        torch.Tensor
            The predicted class probabilities.
        torch.Tensor
            The ground truth labels.
        """
        # TODO: Update data to batch if applicable
        unnormalized_logits = self(data.x, data.edge_index)[:data.batch_size]
        preds = unnormalized_logits.softmax(dim=-1) # predicted class probabilities
        labels = data.y[:data.batch_size]
        return unnormalized_logits, preds, labels


    def training_step(self,
                      loss: torch.Tensor,
                      preds: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """
        Training step for the model.

        Parameters
        ----------
        loss : torch.Tensor
            The computed loss.
        preds : torch.Tensor
            The predicted class probabilities.
        labels : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The training accuracy.

        Notes
        -----
        Throws an error if the loss is not computed by the child class. The default implementation logs the training loss, computes the training accuracy, and logs the training accuracy.
        """
        assert loss is not None, 'Loss not computed'

        self.log(name='train_loss',
                 value=loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        self.train_acc(preds, labels)
        self.log(name='train_acc',
                 value=self.train_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)


    def validation_step(self, data) -> None:
        """
        Validation step for the model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Notes
        -----
        This method may be overridden by subclasses to define the validation step. The default implementation computes the validation accuracy and logs it.
        """
        _, preds, labels = self.common_step(data)

        self.val_acc(preds, labels)
        self.log(name='val_acc',
                 value=self.val_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)


    def test_step(self, data) -> None:
        """
        Test step for the model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Notes
        -----
        This method may be overridden by subclasses to define the test step. The default implementation computes the test accuracy and logs it.
        """
        _, preds, labels = self.common_step(data)

        self.test_acc(preds, labels)
        self.log(name='test_acc',
                 value=self.test_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns
        -------
        torch.optim.Optimizer
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