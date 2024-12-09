"""
Consider a GraphSAGE model with 2 layers. The forward method of the model returns h_v^1 as the output of the first layer and h_v^2. From Step 5 (Algorithm 1) of the paper, the output of the first layer is computed as follows:
`h_v^1 = ReLU(W_0 * CONCAT(x_v, sum_{u \in N(v)} x_u))`

The output of the second layer is computed as follows:
`h_v^2 = W_1 * CONCAT(h_v^1, sum_{u \in N(v)} h_u^1)`

Setting number of features = in_channels, internal dimension = hidden_channels, and number of classes = out_channels, Pytorch Geometric implements a 2-layer GraphSAGE model by using 2 SAGEConv modules.

If we set the layers as follows (and Jumping Knowledge is not used):
Layer 1 = SAGEConv(in_channels, hidden_channels)
Layer 2 = SAGEConv(hidden_channels, out_channels)
Then, internally, the forward of GraphSAGE applies the following operations:
Layer 1:
    - Forward of SAGEConv Layer 1:
        - Message passing step: in_channels -> in_channels
        - Linear transformation (P_0): in_channels -> hidden_channels
    - RELU activation
    - Dropout
Layer 2:
    - Forward of SAGEConv Layer 2:
        - Message passing step: hidden_channels -> hidden_channels
        - Linear transformation (P_1): hidden_channels -> out_channels
    (No activation function is applied in the last layer)
Unfortunately, this means we cannot access the internal node embeddings of the GraphSAGE model. To access the internal node embeddings, we need to separate the encoder from the predictor. We can do this by using the GraphSAGE encoder from Pytorch Geometric and applying the final linear transformation in the predictor layer manually.

Our GraphSAGE model will have the following components:
- GraphSAGE encoder: Pytorch Geometric's SAGEConv module
- Predictor: A linear layer that applies the final linear transformation

More specifically, we do the following:
GraphSAGE encoder:
Layer 1 = SAGEConv(in_channels, hidden_channels)
Layer 2 = SAGEConv(hidden_channels, hidden_channels)
Internally, this applies the following operations:
Layer 1:
    - Forward of SAGEConv Layer 1:
        - Message passing step: in_channels -> in_channels
        - Linear transformation (P_0): in_channels -> hidden_channels
    - RELU activation
    - Dropout
Layer 2:
    - Forward of SAGEConv Layer 2:
        - Message passing step: hidden_channels -> hidden_channels
        - Linear transformation (P_3): hidden_channels -> hidden_channels
    (No activation function is applied in the last layer)

Predictor:
- Linear transformation (P_4): hidden_channels -> out_channels

What has changed is:
- The forward of the encoder now returns the internal node embeddings which are first convolved by the SAGEConv Layer 2 and translated by the linear transformation P_3.
- The forward of the predictor then applies the final linear transformation P_4 to the internal node embeddings to get the unnormalized logits of the model.

Because P_3 and P_4 are both learnable parameters of the model, they will be updated during training. In matrix form, P_3 * P_4 will have the same dimensions as P_1 which is the linear transformation in the last layer of the Pytorch Geometric's GraphSAGE model. Mathematically, this is the same as applying the linear transformation P_1 in the last layer of the Pytorch Geometric's GraphSAGE model. The only downside is that we need to maintain an extra matrix of trainable parameters which impacts the memory usage of the model.
"""


import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GraphSAGE as SAGE_Encoder
from typing import List, Union, Callable

from .base_model import BaseModel


class GraphSAGE(BaseModel):
    def __init__(
            self,
            name: str = 'GraphSAGE',
            in_channels: int = None,
            out_channels: int = None,
            encoder_name: str = 'SAGE_Encoder',
            predictor_name: str = 'Linear',
            hidden_channels: int = 256,
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
        Initializes the GraphSAGE model.

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
        - predictor_name: str
            The name of the predictor module.
        - hidden_channels: int
            The number of hidden features.
        - num_layers: int
            The number of GraphSAGE encoder layers.
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
        super(GraphSAGE, self).__init__(
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

        # Initialize GraphSAGE model from Pytorch Geometric as the encoder
        # The out_channels parameter is not passed to the SAGE_Encoder (i.e. it is set to None) so that we can separate the encoder from the predictor.
        self.encoder = SAGE_Encoder(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            act_first=act_first,
                            act=activation,
                            dropout=dropout,
                            norm=norm
                        )

        # Instead, we apply this final linear transformation in the predictor module manually to have access to the internal node embeddings via the `embed` function.
        self.predictor = nn.Linear(hidden_channels, out_channels)


    @torch.no_grad()
    def embed(
            self,
            subgraph_loader: torch_geometric.data.DataLoader
        ) -> torch.Tensor:
        """
        Computes the internal node embeddings of the GraphSAGE model. These embeddings are of dimension hidden_channels. subgraph_loader is a torch_geometric.data.DataLoader object that may contain the entire graph or a subgraph induced by a batch of nodes.

        Parameters
        ----------
        - subgraph_loader: torch_geometric.data.DataLoader
            The graph data loader. This maybe the full graph or a batch of nodes.

        Returns
        -------
        - torch.Tensor
            The internal node embeddings.

        Notes
        -----
        The input to this method is named `graph_loader` and not `batch_loader` because it may be used to obtain an encoding for any subset of nodes.
        """
        node_embeddings = subgraph_loader.data.x.to(self.device)
        for i in range(self.encoder.num_layers):
            hs = []
            for batch in subgraph_loader:
                h = node_embeddings[batch.n_id].to(self.device)
                h = self.encoder.inference_per_layer(
                        i,
                        h,
                        batch.edge_index.to(self.device),
                        batch.batch_size
                    )
                hs.append(h.to(self.device))

            node_embeddings = torch.cat(hs, dim=0)
        return node_embeddings


    @torch.no_grad()
    def inference(
            self,
            graph_loader: torch_geometric.data.DataLoader
        ) -> torch.Tensor:
        """
        Computes the unnormalized logits of the GraphSAGE model. These logits are of dimension output_channels. This method is used for inference on the full graph.

        Parameters
        ----------
        - graph_loader: torch_geometric.data.DataLoader
            The graph data loader.

        Returns
        -------
        - torch.Tensor
            The unnormalized logits.
        """
        # Compute the internal node embeddings without gradients
        node_embeddings = self.embed(graph_loader)

        # forward pass through the predictor
        unnormalized_logits = self.predictor(node_embeddings)

        return unnormalized_logits


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

        # execute the forward of the GraphSAGE model

        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        unnormalized_logits_batch = self(
                                        train_batch.x,
                                        train_batch.edge_index,
                                    )[:batch_size]

        # prepare dictionary of data required for computing loss
        train_loss_data = {
                        'logits': unnormalized_logits_batch,
                        'labels': train_batch.y[:batch_size],
                        }

        # compute train loss
        train_loss = self.criterion(
                        loss_data=train_loss_data,
                        curr_batch_size=batch_size
                        )

        # compute the predicted class probabilities (normalized logits)
        preds_batch = unnormalized_logits_batch.softmax(dim=-1)

        # compute the training accuracy
        self.train_acc(preds_batch, train_batch.y[:batch_size])

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
            # execute the forward of the GraphSAGE model
            unnormalized_logits_batch = self(
                                            val_batch.x,
                                            val_batch.edge_index
                                        )[:batch_size]

        elif self.inference_mode == 'layer-wise':
            unnormalized_logits_batch = self.val_logits[val_batch.n_id[:batch_size]]

        # prepare dictionary of data required for computing loss
        val_loss_data = {
                        'logits': unnormalized_logits_batch,
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
        self.val_acc(preds_batch, val_batch.y[:batch_size])

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
            unnormalized_logits_batch = self(
                                            test_batch.x,
                                            test_batch.edge_index
                                        )[:batch_size]

        elif self.inference_mode == 'layer-wise':
            unnormalized_logits_batch = self.test_logits[test_batch.n_id[:batch_size]]

        # compute the predicted class probabilities (normalized logits)
        preds_batch = unnormalized_logits_batch.softmax(dim=-1)

        # compute the test accuracy
        self.test_acc(preds_batch, test_batch.y[:batch_size])

        # log the test loss and accuracy
        self.log_metrics(
                mode='test',
                loss_value=None,
                acc_value=self.test_acc,
                curr_batch_size=batch_size,
            )

        return self.test_acc