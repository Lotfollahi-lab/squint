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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
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
                 task: str = 'multiclass',
                 task_kwargs: dict = {},
                 **kwargs):
        """
        Initializes the GraphSAGE model.

        Parameters
        ----------
        in_channels : int
            The number of input features.
        out_channels : int
            The number of output features.
        hidden_channels : int
            The number of hidden features.
        num_layers : int
            The number of GraphSAGE encoder layers.
        act_first : bool
            Whether to apply the activation function before normalization.
        activation : str or callable or None
            The activation function to use.
        norm : str or callable or None
            The normalization function to use.
        dropout : float
            The dropout probability.
        lr : float
            The learning rate.
        weight_decay : float
            The weight decay.
        optimizer_name : str
            The optimizer name.
        loss_names : list of str
            The loss function names.
        loss_kwargs : dict
            Keyword arguments for the loss functions.
        task : str
            The task type.
        task_kwargs : dict
            Keyword arguments for the task.
        """
        # Initialize the BaseModel class
        super().__init__(in_channels=in_channels,
                            out_channels=out_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            dropout=dropout,
                            lr=lr,
                            weight_decay=weight_decay,
                            optimizer_name=optimizer_name,
                            loss_names=loss_names,
                            loss_kwargs=loss_kwargs,
                            task=task,
                            task_kwargs=task_kwargs,
                            **kwargs)

        # Initialize GraphSAGE model from Pytorch Geometric as the encoder
        # The out_channels parameter is not passed to the SAGE_Encoder (i.e. it is set to None) so that we can separate the encoder from the predictor.
        self.encoder = SAGE_Encoder(in_channels=in_channels,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    act_first=act_first,
                                    act=activation,
                                    dropout=dropout,
                                    norm=norm)

        # Instead, we apply this final linear transformation in the predictor module manually to have access to the internal node embeddings via the `embed` function.
        self.predictor = nn.Linear(hidden_channels, out_channels)


    def forward(self,
                batch_x: torch.Tensor,
                batch_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE model. This is a composition of the forward pass of the encoder and the predictor. The batch of nodes may be the entire set of nodes in the graph or a subset of nodes.

        Parameters
        ----------
        batch_x : torch.Tensor
            The input features of the batch of nodes.
        batch_edge_index : torch.Tensor
            The edge index tensor of the batch of nodes.

        Returns
        -------
        torch.Tensor
            The unnormalized logits of the model.
        """
        # calls the forward method of the GraphSAGE encoder
        batch_node_embeddings = self.encoder(batch_x, batch_edge_index)
        # applies linear transformation on the internal node embeddings
        unnormalized_logits = self.predictor(batch_node_embeddings)
        return unnormalized_logits


    @torch.no_grad()
    def embed(self,
              subgraph_loader: torch_geometric.data.DataLoader) -> torch.Tensor:
        """
        Computes the internal node embeddings of the GraphSAGE model. These embeddings are of dimension hidden_channels. subgraph_loader is a torch_geometric.data.DataLoader object that may contain the entire graph or a subgraph induced by a batch of nodes.

        Parameters
        ----------
        subgraph_loader : torch_geometric.data.DataLoader
            The graph data loader. This maybe the full graph or a batch of nodes.

        Returns
        -------
        torch.Tensor
            The internal node embeddings.

        Notes
        -----
        The input to this method is named `graph_loader` and not `batch_loader` to .
        """
        node_embeddings = self.encoder.inference(subgraph_loader)
        return node_embeddings


    @torch.no_grad()
    def inference(self,
                  graph_loader: torch_geometric.data.DataLoader) -> torch.Tensor:
        """
        Computes the unnormalized logits of the GraphSAGE model. These logits are of dimension output_channels. This method is used for inference on the full graph.

        Parameters
        ----------
        graph_loader : torch_geometric.data.DataLoader
            The graph data loader.

        Returns
        -------
        torch.Tensor
            The unnormalized logits.
        """
        node_embeddings = self.embed(graph_loader)
        unnormalized_logits = self.predictor(node_embeddings)
        return unnormalized_logits


    def training_step(self,
                      data: torch_geometric.data.Data) -> torch.Tensor:
        """
        Training step for the GraphSAGE model. Overrides the training_step method of the BaseModel class to prepare data required for computing loss. Calls super().training_step to log the training loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        # collect data required for computing the loss
        unnormalized_logits, preds, labels = self.common_step(data)

        # prepare dictionary of data at current step for computing loss
        loss_data = {'logits': unnormalized_logits,
                     'labels': labels}

        # compute loss
        loss = self.criterion(loss_data)

        # log the training loss and accuracy
        super().training_step(loss, preds, labels)

        return loss