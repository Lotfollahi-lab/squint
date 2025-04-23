"""
This file implements the Vanilla GNN Model.

The Vanilla GNN Model is a simple GNN model that comprises of the following components:
- Encoder: A series of Linear layers followed by a GraphSAGE, GATv2, or GIN module.
- Attribute Decoder: A Linear or LinearSoftmax decoder.
- Predictor: A Linear predictor.

In addition, this provides the option to log the mean pairwise cosine similarity between the original attributes, decoded attributes, and GNN embeddings, and the Pearson correlation between the original and decoded node attributes at the end of each training epoch.
"""
from typing import List, Union, Callable, Literal, Dict

import torch
import torch_geometric
from torch_geometric.nn.dense.linear import Linear

from .base_model import BaseModel
from ..utils import metrics


class MLP(BaseModel):
    def __init__(
            self,
            model_name: Literal['MLP'] = 'MLP',
            encoder_name: Literal['Linear'] = 'Linear',
            attribute_decoder_name: Literal['Linear', 'LinearSoftmax'] = 'Linear',
            predictor_name: Literal['Linear'] = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            log_similarity_stats: bool = False,
            log_pearson_correlation: bool = False,
            num_linear_layers: int = 1,
            hidden_channels: List[int] | int = 500,
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
        Initializes a Vanilla GNN model. Currently, this class supports GraphSAGE, GATv2, and GIN encoders.

        Parameters
        ----------
        - model_name: Literal['MLP']
            The name of the model.
        - encoder_name: Literal['Linear']
            The name of the encoder module.
        - attribute_decoder_name: Literal['Linear', 'LinearSoftmax']
            The name of the attribute decoder module.
        - predictor_name: Literal['Linear']
            The name of the predictor module.
        - log_similarity_stats: bool
            Whether to log the similarity statistics.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation.

        - in_channels: int
            The number of input features.
        - out_channels: int
            The number of output features.

        - num_linear_layers: int
            The number of linear layers to use before the GraphSAGE encoder layers.
        - hidden_channels: List[int] | int
            The number of hidden features.
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
        print(f"Initializing a {model_name} Model with the following components:")

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

        # Initialize a MLP module as the encoder.
        self.encoder = Linear(
            in_channels=in_channels,
            out_channels=hidden_channels,
            weight_initializer=init_method,
        )

        # Initialize the attribute decoder.
        self.attribute_decoder = self._init_attribute_decoder(
            attribute_decoder_name=attribute_decoder_name,
            in_channels=hidden_channels,
            out_channels=in_channels,
            init_method=init_method,
        )
        print(f"2. Attribute Decoder: {attribute_decoder_name} that reconstructs {hidden_channels} latent features to {in_channels} input features.")

        # Initialize the predictor.
        # Currently, the predictor is hardcoded to be a simple linear layer.
        self.predictor = Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            weight_initializer=init_method,
        )
        print(f"3. Predictor: Linear layer that transforms {hidden_channels} hidden features to {out_channels} output features.")


    def forward(
            self,
            batch_x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE model.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features.

        Returns
        -------
        - h_encoder: torch.Tensor
            The latent node embeddings from the MLP encoder.
        - xhat: torch.Tensor
            The decoded node attributes.
        - unnormalized_logits: torch.Tensor
            The unnormalized logits.
        """
        # forward pass through the MLP encoder
        h_encoder = self.encoder(
                        batch_x,
                    )

        # decode the latent node embeddings to recover the node attributes
        xhat = self.attribute_decoder(
                            x=h_encoder,
                            read_depth=batch_x.sum(dim=-1)
                        )

        # forward pass through the predictor
        unnormalized_logits = self.predictor(h_encoder)

        return h_encoder, \
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
        unnormalized_logits_batch = self(train_batch.x)

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
        unnormalized_logits_batch = self(val_batch.x)

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
        unnormalized_logits_batch = self(test_batch.x)

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
        H_encoder = []
        X_hat = []

        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_type.append(batch.y[:batch_size])
            Y_niche_type.append(batch.y_niche_types[:batch_size])

            h_encoder, \
            xhat_batch, \
            _ = self(batch.x.to(self.device))

            H_encoder.append(h_encoder[:batch_size])
            X_hat.append(xhat_batch[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_type = torch.cat(Y_cell_type, dim=0)
        Y_niche_type = torch.cat(Y_niche_type, dim=0)
        H_encoder = torch.cat(H_encoder, dim=0)
        X_hat = torch.cat(X_hat, dim=0)

        return X, \
                Y_cell_type, \
                Y_niche_type, \
                H_encoder, \
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
        H_encoder, \
        X_hat = self.inference()

        train_epoch_end_stats = {}

        if self.log_similarity_stats:
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X, 'X')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(H_encoder, 'H_encoder')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X_hat, 'X_hat')
            )

        if self.log_pearson_correlation:
            pearson = metrics.pearson_correlation(
                            X.cpu().numpy(),
                            X_hat.cpu().numpy(),
                            compare_genes=False,
                            mean=True,
                        )
            train_epoch_end_stats['pearson_correlation'] = pearson
        return train_epoch_end_stats
