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
- The forward of the encoder now returns the latent node embeddings which are first convolved by the SAGEConv Layer 2 and translated by the linear transformation P_3.
- The attribute decoder takes the latent node embeddings as input and outputs an estimate of the original node attributes.
- The forward of the predictor applies the final linear transformation P_4 to the latent node embeddings to get the unnormalized logits of the model.

Because P_3 and P_4 are both learnable parameters of the model, they will be updated during training. In matrix form, P_3 * P_4 will have the same dimensions as P_1 which is the linear transformation in the last layer of the Pytorch Geometric's GraphSAGE model. Mathematically, this is the same as applying the linear transformation P_1 in the last layer of the Pytorch Geometric's GraphSAGE model. The only downside is that we need to maintain an extra matrix of trainable parameters which impacts the memory usage of the model.
"""
from typing import List, Union, Callable, Literal, Dict

import torch
import torch.nn as nn
import torch_geometric
from ..modules.sage_conv import SAGEConv_Module as GraphSAGE_Encoder

from .base_model import BaseModel
from ..utils import metrics


class GraphSAGE(BaseModel):
    def __init__(
            self,
            model_name: str = 'GraphSAGE',
            encoder_name: str = 'SAGE_Encoder',
            attribute_decoder_name: Literal['Linear', 'LinearSoftmax'] = 'Linear',
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
        - attribute_decoder_name: Literal['Linear', 'LinearSoftmax']
            The name of the attribute decoder module.
        - predictor_name: str
            The name of the predictor module.
        - log_similarity_stats: bool
            Whether to log the similarity statistics.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation.

        - in_channels: int
            The number of input features.
        - out_channels: int
            The number of output features.

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
            attribute_decoder_name=attribute_decoder_name,
            predictor_name=predictor_name,
            log_similarity_stats=log_similarity_stats,
            log_pearson_correlation=log_pearson_correlation,
            in_channels=in_channels,
            out_channels=out_channels,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            loss_names=loss_names,
            loss_kwargs=loss_kwargs,
        )

        # Initialize SAGEConv_Module as the encoder.
        # The out_channels parameter is not passed to the SAGEConv_Module (i.e. it is set to None) so that we can separate the encoder from the predictor.
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

        # Initialize the attribute decoder.
        self.attribute_decoder = self._init_attribute_decoder(
            attribute_decoder_name=attribute_decoder_name,
            in_channels=hidden_channels,
            out_channels=in_channels
        )

        # Initialize the predictor.
        # Currently, the predictor hardcoded to be a simple linear layer.
        self.predictor = nn.Linear(
                            in_features=hidden_channels,
                            out_features=out_channels
                        )

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
        - h_latent: torch.Tensor
            The latent node embeddings from the SAGEConv layers.
        - xhat: torch.Tensor
            The decoded node attributes.
        - unnormalized_logits: torch.Tensor
            The unnormalized logits.
        """
        # forward pass through the graph encoder
        h_latent = self.encoder(
                        batch_x,
                        batch_edge_index
                    )

        # decode the latent node embeddings to recover the node attributes
        xhat = self.attribute_decoder(
                            x=h_latent,
                            read_depth=batch_x.sum(dim=-1)
                        )

        # forward pass through the predictor
        unnormalized_logits = self.predictor(h_latent)

        return h_latent, \
                xhat, \
                unnormalized_logits


    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
        ) -> torch.Tensor:
        """
        Definition of a single training step of the GraphSAGE model on the current batch of nodes received from the training dataloader at the current training epoch.

        Parameters
        ----------
        - train_batch: torch_geometric.data.Data
            The input train data (batch of nodes).

        Returns
        -------
        - train_loss: torch.Tensor
            The computed loss for this batch.
        """
        # execute the forward of the GraphSAGE model
        _, \
        xhat_batch, \
        unnormalized_logits_batch = self(
                                        train_batch.x,
                                        train_batch.edge_index,
                                    )

        # prepare dictionary of data required for computing loss
        # Slicing via batch_size is necessary because the Loader places the target nodes, i.e. nodes in the current batch, at the start of the batch.
        # IDs of the target nodes are accessible via train_batch.input_id or train_batch.n_id[:batch_size].
        # train_batch.n_id contains the IDs of all the target nodes and their sampled neighbors.
        batch_size = train_batch.batch_size
        train_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': train_batch.y[:batch_size],
                        'pred_attr': xhat_batch[:batch_size],
                        'target_attr': train_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        }

        train_loss = self.common_step(
            batch_loss_data=train_loss_data,
            batch_size=batch_size,
            mode='train',
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
        # execute the forward of the GraphSAGE model
        _, \
        xhat_batch, \
        unnormalized_logits_batch = self(
                                        val_batch.x,
                                        val_batch.edge_index
                                    )

        # prepare dictionary of data required for computing loss
        batch_size = val_batch.batch_size
        val_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
                        'pred_attr': xhat_batch[:batch_size],
                        'target_attr': val_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        }

        val_loss = self.common_step(
            batch_loss_data=val_loss_data,
            batch_size=batch_size,
            mode='val',
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
        # execute the forward of the GraphSAGE model
        _, \
        _, \
        unnormalized_logits_batch = self(
                                        test_batch.x,
                                        test_batch.edge_index
                                    )

        # prepare dictionary of data required for computing accuracy
        batch_size = test_batch.batch_size
        test_acc_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': test_batch.y[:batch_size],
                        }

        test_acc = self.common_step(
            batch_loss_data=test_acc_data,
            batch_size=batch_size,
            mode='test',
        )

        return test_acc


    @torch.no_grad()
    def inference(self):
        X = []
        Y_cell_type = []
        Y_niche_type = []
        H_latent = []
        X_hat = []

        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_type.append(batch.y[:batch_size])
            Y_niche_type.append(batch.y_niche_types[:batch_size])

            h_latent, \
            xhat_batch, \
            _ = self(
                        batch.x.to(self.device),
                        batch.edge_index.to(self.device)
                    )

            H_latent.append(h_latent[:batch_size])
            X_hat.append(xhat_batch[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_type = torch.cat(Y_cell_type, dim=0)
        Y_niche_type = torch.cat(Y_niche_type, dim=0)
        H_latent = torch.cat(H_latent, dim=0)
        X_hat = torch.cat(X_hat, dim=0)

        return X, \
                Y_cell_type, \
                Y_niche_type, \
                H_latent, \
                X_hat


    @torch.no_grad()
    def compute_train_epoch_stats(self) -> Dict:
        """
        If the log_similarity_stats flag is set to True, compute statistics for original and decoded node attributes, and latent node embeddings at the end of each training epoch.
        If the log_pearson_correlation flag is set to True, compute the Pearson correlation between original and decoded attributes at the end of each training epoch.

        Returns
        -------
        - train_epoch_end_stats: dict
            Dictionary containing the computed statistics.
        """
        X, \
        _, \
        _, \
        H_latent, \
        X_hat = self.inference()

        train_epoch_end_stats = {}

        if self.log_similarity_stats:
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X, 'X')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(H_latent, 'H_latent')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X_hat, 'X_hat')
            )

        if self.log_pearson_correlation:
            train_epoch_end_stats['pearson_correlation'] = metrics.pearson_correlation(
                                                            X.cpu().numpy(),
                                                            X_hat.cpu().numpy(),
                                                            compare_genes=False,
                                                            mean=True,
                                                        )

        return train_epoch_end_stats
