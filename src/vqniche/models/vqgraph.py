"""

"""

import torch
import torch.nn as nn
import torch_geometric
from ..encoders.vqgraph_encoder import VQGraph_Encoder
from typing import List, Union, Callable

from .base_model import BaseModel


class VQGraph(BaseModel):
    def __init__(
            self,
            name: str = 'VQGraph',
            in_channels: int = None,
            out_channels: int = None,
            encoder_name: str = 'VQGraph_Encoder',
            graphconv_layer_name: str = 'SAGEConv',
            predictor_name: str = 'Linear',
            hidden_channels: int = 64,
            num_layers: int = 2,
            act_first: bool = True,
            activation: Union[str, Callable, None] = "relu",
            norm: Union[str, Callable, None] = None,
            dropout: float = 0.5,
            lr: float = 0.01,
            weight_decay: float = 0.0,
            optimizer_name: str = 'adam',
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'none'},
            task_name: str = 'multiclass',
            task_kwargs: dict = {},
            inference_mode: str = 'batch-wise',
            **kwargs
        ):
        """
        Initializes the VQGraph model.

        Parameters
        ----------
        - name: str
            The name of the model.
        - in_channels: int
            The number of input features.
        - out_channels: int
            The number of output features.
        - encoder_name: str
            The name of the encoder module.
        - graphconv_layer_name: str
            The name of the graph convolutional layer.
        - predictor_name: str
            The name of the predictor module.
        - hidden_channels: int
            The number of hidden features.
        - num_layers: int
            The number of VQGraph encoder layers.
        - act_first: bool
            Whether to apply the activation function before normalization.
        - activation: str or callable or None
            The activation function to use.
        - norm: str or callable or None
            The normalization function to use.
        - dropout: float
            The dropout probability.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.
        - optimizer_name: str
            The optimizer name.
        - loss_names: list of str
            The loss function names.
        - loss_kwargs: dict
            Keyword arguments for the loss functions.
        - task_name: str
            The task type.
        - task_kwargs: dict
            Keyword arguments for the task.
        - inference_mode: str
            The inference mode. Choose from 'batch-wise' or 'layer-wise'.
        - kwargs: dict
            Additional keyword arguments.
        """
        # Initialize the BaseModel class
        super(VQGraph, self).__init__(
                        name=name,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        encoder_name=encoder_name,
                        predictor_name=predictor_name,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers,
                        dropout=dropout,
                        lr=lr,
                        weight_decay=weight_decay,
                        optimizer_name=optimizer_name,
                        loss_names=loss_names,
                        loss_kwargs=loss_kwargs,
                        task_name=task_name,
                        task_kwargs=task_kwargs,
                        inference_mode=inference_mode,
                        **kwargs
                    )

        # Initialize VQGraph encoder module.
        # The out_channels parameter is not passed to the VQGraph_Encoder (i.e. it is set to None) so that we can separate the encoder from the predictor.
        self.encoder = VQGraph_Encoder(
                            graphconv_layer_name=graphconv_layer_name,
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            act_first=act_first,
                            activation=activation,
                            dropout=dropout,
                            norm=norm,
                        )

        # Instead, we apply this final linear transformation in the predictor module manually to have access to the internal node embeddings via the `embed` function.
        self.predictor = nn.Linear(hidden_channels, out_channels)


    def forward(
        self,
        batch_x: torch.Tensor,
        batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the VQGraph model.
        """
        # execute the forward of the VQGraph_Encoder model
        h_pre_vq_conv, \
        h_vq, \
        indices, \
        dist, \
        codebook_embeddings, \
        h_node, \
        h_edge, \
        h_post_vq_conv \
            = self.encoder(
                            batch_x,
                            batch_edge_index
                        )

        # Apply the predictor to the VQ-encoded node embeddings.
        unnormalized_logits_batch = self.predictor(h_post_vq_conv)

        return h_pre_vq_conv, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings, \
            h_node, \
            h_edge, \
            h_post_vq_conv, \
            unnormalized_logits_batch


    # NOTE: Add the following method to the BaseModel class.
    # @torch.no_grad()
    # def embed(

    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
            batch_idx: int,
        ) -> torch.Tensor:
        """
        Definition of a single training step of the GraphSAGE model on the current batch of nodes received from the training dataloader at the current training epoch.

        Parameters
        ----------
        - batch: torch_geometric.data.Data
            The input train data (batch of nodes).
        - batch_idx: int
            The index of the current batch of data.

        Returns
        -------
        - torch.Tensor
            The computed loss for this batch.
        """
        batch_size = train_batch.batch_size

        # execute the forward of the VQGraph model
        h_pre_vq_conv, \
        h_vq, \
        indices, \
        dist, \
        codebook_embeddings, \
        h_node, \
        h_edge, \
        h_post_vq_conv, \
        unnormalized_logits_batch \
            = self(
                    train_batch.x,
                    train_batch.edge_index,
                )

        # prepare dictionary of data required for computing loss
        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        train_loss_data = {
                        'h_pre_vq_conv': h_pre_vq_conv[:batch_size],
                        'h_vq': h_vq[:batch_size],
                        'indices': indices[:batch_size],
                        'dist': dist[:batch_size],
                        'codebook_embeddings': codebook_embeddings[:batch_size],
                        'h_node': h_node[:batch_size],
                        'h_edge': h_edge[:batch_size],
                        'h_post_vq_conv': h_post_vq_conv[:batch_size],
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': train_batch.y[:batch_size],
                        'batch_edge_index': train_batch.edge_index,
                        }

        # compute train loss
        train_loss = self.criterion(
                        loss_data=train_loss_data,
                        curr_batch_size=batch_size
                        )

        # compute the predicted class probabilities (normalized logits)
        preds_batch = unnormalized_logits_batch.softmax(dim=-1)

        # compute the training accuracy
        self.train_acc(preds_batch[:batch_size], train_batch.y[:batch_size])

        # log the training loss and accuracy
        self.log_metrics(
                mode='train',
                loss_value=train_loss,
                acc_value=self.train_acc,
                curr_batch_size=batch_size,
            )

        return train_loss


    def validation_step(
            self,
            val_batch: torch_geometric.data.Data
        ) -> torch.Tensor:
        """
        Definition of a single validation step of the GraphSAGE model on the current batch of nodes received from the validation dataloader at the current training epoch.

        Parameters
        ----------
        - val_batch: torch_geometric.data.Data
            The input validation data (batch of nodes).

        Returns
        -------
        - torch.Tensor
            The computed loss for this batch.
        """
        batch_size = val_batch.batch_size

        if self.inference_mode == 'batch-wise':
            h_pre_vq_conv, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings, \
            h_node, \
            h_edge, \
            h_post_vq_conv, \
            unnormalized_logits_batch \
                = self(
                        val_batch.x,
                        val_batch.edge_index,
                    )

        elif self.inference_mode == 'layer-wise':
            raise NotImplementedError("Layer-wise inference is not supported for validation step.")

        # prepare dictionary of data required for computing loss
        val_loss_data = {
                        'h_pre_vq_conv': h_pre_vq_conv[:batch_size],
                        'h_vq': h_vq[:batch_size],
                        'indices': indices[:batch_size],
                        'dist': dist[:batch_size],
                        'codebook_embeddings': codebook_embeddings[:batch_size],
                        'h_node': h_node[:batch_size],
                        'h_edge': h_edge[:batch_size],
                        'h_post_vq_conv': h_post_vq_conv[:batch_size],
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
                        }

        # compute validation loss
        val_loss = self.criterion(
                        loss_data=val_loss_data,
                        curr_batch_size=batch_size
                        )

        # compute the predicted class probabilities (normalized logits)
        preds_batch = unnormalized_logits_batch.softmax(dim=-1)

        # compute the validation accuracy
        self.val_acc(preds_batch[:batch_size], val_batch.y[:batch_size])

        # log the validation loss and accuracy
        self.log_metrics(
                mode='val',
                loss_value=val_loss,
                acc_value=self.val_acc,
                curr_batch_size=batch_size,
            )

        return val_loss


    def test_step(
            self,
            test_batch: torch_geometric.data.Data
        ) -> torch.Tensor:
        """
        Definition of a single test step of the GraphSAGE model on the current batch of nodes received from the test dataloader at the current training epoch.

        Parameters
        ----------
        - test_batch: torch_geometric.data.Data
            The input test data (batch of nodes).

        Returns
        -------
        - torch.Tensor
            The computed loss for this batch.
        """
        batch_size = test_batch.batch_size

        if self.inference_mode == 'batch-wise':
            # execute the forward of the GraphSAGE model
            _, \
            _, \
            _, \
            _, \
            _, \
            _, \
            _, \
            _, \
            unnormalized_logits_batch \
                = self(
                        test_batch.x,
                        test_batch.edge_index
                    )

        elif self.inference_mode == 'layer-wise':
            raise NotImplementedError("Layer-wise inference is not supported for test step.")

        # compute the predicted class probabilities (normalized logits)
        preds_batch = unnormalized_logits_batch.softmax(dim=-1)

        # compute the test accuracy
        self.test_acc(preds_batch[:batch_size], test_batch.y[:batch_size])

        # log the test loss and accuracy
        self.log_metrics(
                mode='test',
                loss_value=None,
                acc_value=self.test_acc,
                curr_batch_size=batch_size,
            )

        return self.test_acc