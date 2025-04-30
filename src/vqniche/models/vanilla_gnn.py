"""
This file implements the Vanilla GNN Model.

The Vanilla GNN Model is a simple GNN model that comprises of the following components:
- Encoder: A series of Linear layers followed by a GraphSAGE, GATv2, or GIN module.
- Attribute Decoder: A Linear or LinearSoftmax decoder.
- Predictor: A Linear predictor.

In addition, this provides the option to log the mean pairwise cosine similarity between the original attributes, decoded attributes, and GNN embeddings, and the Pearson correlation between the original and decoded node attributes at the end of each training epoch.
"""
from typing import Literal, Dict

import torch
import torch_geometric
from .base_model import BaseModel
from ..encoders.vanilla_gnn_encoder import VanillaGNN_Encoder
from ..utils import metrics


class VanillaGNN(BaseModel):
    def __init__(
            self,
            model_name: Literal['GraphSAGE', 'GATv2', 'GIN'] = 'GraphSAGE',
            encoder_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv',
            attribute_decoder_name: Literal['Linear', 'LinearSoftmax'] = 'Linear',
            predictor_name: Literal['Linear'] = 'Linear',
            log_similarity_stats: bool = False,
            log_pearson_correlation: bool = False,
            in_channels: int = None,
            out_channels: int = None,
            encoder_params: dict = {},
            attribute_decoder_params: dict = {},
            optimizer_params: dict = {},
            loss_params: dict = {},
        ):
        """
        Initializes a Vanilla GNN model. Currently, this class supports GraphSAGE, GATv2, and GIN encoders.

        Parameters
        ----------
        - model_name: Literal['GraphSAGE', 'GATv2', 'GIN']
            The name of the model.
        - encoder_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv']
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

        - encoder_params: dict
            The parameters for the MLP module.
        - attribute_decoder_params: dict
            The parameters for the attribute decoder module.
        - optimizer_params: dict
            The parameters for the optimizer.
        - loss_params: dict
            The parameters for the loss function.
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
            **optimizer_params,
            **loss_params,
        )

        # Initialize a VanillaGNN_Encoder.
        # This module applies either a GNN module or an MLP followed by a GNN module to build latent node embeddings.
        # The out_channels parameter is not passed to the GNN_Module (i.e. it is set to None) to separate the encoder from the predictor.
        self.encoder = VanillaGNN_Encoder(
                            in_channels=in_channels,
                            gnn_name=encoder_name,
                            **encoder_params
                        )
        print(f"1. Encoder: {self.encoder.mlp_layers} Linear layer(s) followed by {self.encoder.gnn_layers} {self.encoder.gnn_module.gnn_layer_name} layer(s) that transform {in_channels} input features to {self.encoder.dim} hidden features.")

        # Initialize the attribute decoder.
        self.attribute_decoder = self._init_attribute_decoder(
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=attribute_decoder_params,
        )
        print(f"2. Attribute Decoder: {attribute_decoder_name} that decodes {self.encoder.dim} latent features to {in_channels} input features.")

        # Initialize the predictor.
        # Currently, the predictor is hardcoded to be a simple linear layer.
        self.predictor = self._init_predictor(
                            predictor_name=predictor_name,
                            in_channels=self.encoder.dim,
                            out_channels=out_channels,
                        )
        print(f"3. Predictor: Linear layer that transforms {self.encoder.dim} hidden features to {out_channels} dimensional logits.")


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
            Latent node embeddings from the Vanilla GNN Encoder.
        - xhat: torch.Tensor
            Reconstructed node attributes from the Attribute Decoder.
        - unnormalized_logits: torch.Tensor
            Unnormalized logits from the Predictor.

        Notes
        -----
        - h_latent is either the output of the GNN module (if no MLP module is used) or the output of the MLP module followed by the GNN module (if a MLP module is used).
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
                        'pred_attr': xhat_batch[:batch_size],
                        'target_attr': train_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': train_batch.y[:batch_size],
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
                        'pred_attr': xhat_batch[:batch_size],
                        'target_attr': val_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
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
            pearson = metrics.pearson_correlation(
                            X.cpu().numpy(),
                            X_hat.cpu().numpy(),
                            compare_genes=False,
                            mean=True,
                        )
            train_epoch_end_stats['pearson_correlation'] = pearson
        return train_epoch_end_stats
