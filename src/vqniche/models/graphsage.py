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
from typing import List, Union, Callable, Literal, Dict

import torch
import torch.nn as nn
import torch_geometric
from ..modules.sage_conv import SAGEConv_Module as GraphSAGE_Encoder

from .base_model import BaseModel
from ..utils import metrics
from ..modules.linear_softmax_decoder import LinearSoftmax
from ..utils.metrics import compute_pearson_correlation


class GraphSAGE(BaseModel):
    def __init__(
            self,
            model_name: str = 'GraphSAGE',
            encoder_name: str = 'SAGE_Encoder',
            predictor_name: str = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            log_similarity_stats: bool = False,
            log_pearson_correlation: bool = False,
            hidden_channels: int = 256,
            num_layers: int = 2,
            act_first: bool = True,
            activation: Union[str, Callable, None] = "relu",
            norm: Union[str, Callable, None] = None,
            dropout: float = 0.5,
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'none'},
        ):
        """
        Initializes the GraphSAGE model.

        Parameters
        ----------
        - model_name: str
            The name of the model.
        - encoder_name: str
            The name of the encoder module.
        - predictor_name: str
            The name of the predictor module.

        - in_channels: int
            The number of input features.
        - out_channels: int
            The number of output features.
        - log_similarity_stats: bool
            Whether to log the similarity statistics.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation.

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
        - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
            The initialization method to use for the linear transformations in the SAGEConv layers.
            If None, the initialization method is 'kaiming_uniform'.

        - optimizer_name: str
            The optimizer name.
        - lr: float
            The learning rate.
        - weight_decay: float
            The weight decay.

        - loss_names: list of str
            The loss function names.
        - loss_kwargs: dict
            Keyword arguments for the loss functions.
        """
        # Initialize the BaseModel class
        super().__init__(
            model_name=model_name,
            encoder_name=encoder_name,
            predictor_name=predictor_name,
            in_channels=in_channels,
            out_channels=out_channels,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            loss_names=loss_names,
            loss_kwargs=loss_kwargs,
        )

        # Initialize GraphSAGE model from Pytorch Geometric as the encoder
        # The out_channels parameter is not passed to the SAGE_Encoder (i.e. it is set to None) so that we can separate the encoder from the predictor.
        self.encoder = GraphSAGE_Encoder(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            act_first=act_first,
                            activation=activation,
                            norm=norm,
                            dropout=dropout,
                            init_method=init_method
                        )

        self.attribute_decoder = LinearSoftmax(
                name='LinearSoftmax',
                in_channels=hidden_channels,
                out_channels=in_channels
            )

        # Instead, we apply this final linear transformation in the predictor module manually to have access to the internal node embeddings via the `embed` function.
        self.predictor = nn.Linear(
                            in_features=hidden_channels,
                            out_features=out_channels
                        )

        self.log_similarity_stats = log_similarity_stats
        self.log_pearson_correlation = log_pearson_correlation


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE model.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features.
        - batch_edge_index: torch.Tensor
            The edge index.

        Returns
        -------
        - h_encoder: torch.Tensor
            The internal node embeddings.
        - h_attr_decoded: torch.Tensor
            The decoded node attributes.
        - unnormalized_logits: torch.Tensor
            The unnormalized logits.
        """
        # forward pass through the graph encoder
        h_encoder = self.encoder(
                        batch_x,
                        batch_edge_index
                    )

        # decode the node embeddings to recover the node attributes
        h_attr_decoded = self.attribute_decoder(
                            x=h_encoder,
                            read_depth=batch_x.sum(dim=-1)
                        )

        # forward pass through the predictor
        unnormalized_logits = self.predictor(h_encoder)

        return h_encoder, \
                h_attr_decoded, \
                unnormalized_logits


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
        - train_loss: torch.Tensor
            The computed loss for this batch.
        """
        batch_size = train_batch.batch_size

        # execute the forward of the GraphSAGE model

        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        _, \
        h_attr_decoded, \
        unnormalized_logits_batch = self(
                                        train_batch.x,
                                        train_batch.edge_index,
                                    )

        # prepare dictionary of data required for computing loss
        train_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': train_batch.y[:batch_size],
                        'pred_attr': h_attr_decoded[:batch_size],
                        'target_attr': train_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        }

        # compute train loss
        train_loss = self.criterion(
                        loss_data=train_loss_data,
                        curr_batch_size=batch_size
                        )

        # compute train accuracy
        train_acc = metrics.accuracy_score(
                        unnormalized_logits=unnormalized_logits_batch[:batch_size],
                        one_hot_labels=train_batch.y[:batch_size],
                    )

        # log training loss and accuracy
        self.log_metrics(
                mode='train',
                loss_value=train_loss,
                acc_value=train_acc,
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
        - val_loss: torch.Tensor
            The computed loss for this batch.
        """
        batch_size = val_batch.batch_size

        _, \
        h_attr_decoded, \
        unnormalized_logits_batch = self(
                                        val_batch.x,
                                        val_batch.edge_index
                                    )

        # prepare dictionary of data required for computing loss
        val_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
                        'pred_attr': h_attr_decoded[:batch_size],
                        'target_attr': val_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        }

        # compute validation loss
        val_loss = self.criterion(
                        loss_data=val_loss_data,
                        curr_batch_size=batch_size
                        )

        # compute validation accuracy
        val_acc = metrics.accuracy_score(
                        unnormalized_logits=unnormalized_logits_batch[:batch_size],
                        one_hot_labels=val_batch.y[:batch_size],
                    )

        # log validation loss and accuracy
        self.log_metrics(
                mode='val',
                loss_value=val_loss,
                acc_value=val_acc,
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
        - test_loss: torch.Tensor
            The computed loss for this batch.
        """
        batch_size = test_batch.batch_size

        _, \
        _, \
        unnormalized_logits_batch = self(
                                        test_batch.x,
                                        test_batch.edge_index
                                    )

        # compute test accuracy
        test_acc = metrics.accuracy_score(
                        unnormalized_logits=unnormalized_logits_batch[:batch_size],
                        one_hot_labels=test_batch.y[:batch_size],
                    )

        # log test accuracy
        self.log_metrics(
                mode='test',
                loss_value=None,
                acc_value=test_acc,
                curr_batch_size=batch_size,
            )

        return test_acc


    @torch.no_grad()
    def compute_train_epoch_stats(self) -> Dict:
        """
        Compute pairwise similarity statistics for all embeddings.

        Returns
        -------
        - similarity_stats: dict
            Dictionary containing mean and std of pairwise cosine similarities for different embeddings
        """
        train_epoch_end_stats = {}

        if self.log_similarity_stats:
            h_encoder_list = []
            h_attr_decoded_list = []
        if self.log_pearson_correlation:
            X = []
            X_hat = []

        # Iterate through inference dataloader
        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size
            h_encoder, \
            h_attr_decoded, \
            _ = self(
                        batch.x.to(self.device),
                        batch.edge_index.to(self.device)
                    )

            if self.log_similarity_stats:
                h_encoder_list.append(h_encoder[:batch_size])
                h_attr_decoded_list.append(h_attr_decoded[:batch_size])

            if self.log_pearson_correlation:
                X.append(batch.x[:batch_size])
                X_hat.append(h_attr_decoded[:batch_size])

        # Compute statistics for all embeddings
        if self.log_similarity_stats:
            h_encoder = torch.cat(h_encoder_list, dim=0)
            h_attr_decoded = torch.cat(h_attr_decoded_list, dim=0)
            train_epoch_end_stats.update(
                metrics.get_similarity_stats(h_encoder, 'h_encoder')
            )
            train_epoch_end_stats.update(
                metrics.get_similarity_stats(h_attr_decoded, 'h_attr_decoded')
            )

        if self.log_pearson_correlation:
            X = torch.cat(X, dim=0)
            X_hat = torch.cat(X_hat, dim=0)
            pearson_correlation = compute_pearson_correlation(
                            X.cpu().numpy(),
                            X_hat.cpu().numpy(),
                            compare_genes=False,
                            mean=True,
                        )
            train_epoch_end_stats['pearson_correlation'] = pearson_correlation

        return train_epoch_end_stats
