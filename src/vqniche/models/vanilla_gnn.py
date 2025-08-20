"""
This file implements the Vanilla GNN Model.

The Vanilla GNN Model is a simple GNN model that comprises of the following components:
- Encoder: A series of Linear layers followed by a GraphSAGE, GATv2, or GIN module.
- Attribute Decoder: A Linear or LinearSoftmax decoder.
- Predictor: A Linear predictor.

In addition, this provides the option to log the mean pairwise cosine similarity between the original attributes, decoded attributes, and GNN embeddings, and the Pearson correlation between the original and decoded node attributes at the end of each training epoch.
"""
from typing import Literal, List

import torch
import torch_geometric

from .base_model import BaseModel
from ..encoders.vanilla_gnn_encoder import VanillaGNN_Encoder


class VanillaGNN(BaseModel):
    def __init__(
            self,
            model_name: Literal['GraphSAGE', 'GATv2', 'GIN'] = 'GraphSAGE',
            imputation_params: dict = {},
            encoder_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv',
            attribute_decoder_name: Literal['Linear', 'LinearSoftmax'] = 'Linear',
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            predictor_name: Literal['Linear'] = 'Linear',
            in_channels: int = None,
            condition_dim: int = 0,
            out_channels: int = None,
            train_metrics_list: List[str] = [],
            encoder_params: dict = {},
            attribute_decoder_params: dict = {},
            adjacency_decoder_params: dict = {},
            optimizer_params: dict = {},
            loss_params: dict = {},
        ):
        """
        Initializes a Vanilla GNN model. Currently, this class supports GraphSAGE, GATv2, and GIN encoders.

        Parameters
        ----------
        - model_name: Literal['GraphSAGE', 'GATv2', 'GIN']
            The name of the model.
        - imputation_params: dict
            The parameters for the imputation module.
        - encoder_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv']
            The name of the encoder module.
        - attribute_decoder_name: Literal['Linear', 'LinearSoftmax']
            The name of the attribute decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - predictor_name: Literal['Linear']
            The name of the predictor module.

        - in_channels: int
            The number of input features.
        - condition_dim: int
            The number of conditioning features.
        - out_channels: int
            The number of output features.

        - train_log_flags: dict
            The flags for logging metrics during training.

        - encoder_params: dict
            The parameters for the MLP module.
        - attribute_decoder_params: dict
            The parameters for the attribute decoder module.
        - adjacency_decoder_params: dict
            The parameters for the adjacency decoder module.
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
            adjacency_decoder_name=adjacency_decoder_name,
            predictor_name=predictor_name,
            in_channels=in_channels,
            out_channels=out_channels,
            train_metrics_list=train_metrics_list,
            **optimizer_params,
            **loss_params,
        )

        self.imputation_params = imputation_params

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
            in_channels=self.encoder.dim,
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=attribute_decoder_params,
        )
        print(f"2. Attribute Decoder: {attribute_decoder_name} that decodes latent embeddings of dimension {self.encoder.dim} to input features of dimension {in_channels}.")

        # Initialize the adjacency decoder.
        self.adjacency_decoder = self._init_adjacency_decoder(
            in_channels=self.encoder.dim,
            adjacency_decoder_name=adjacency_decoder_name,
            **adjacency_decoder_params,
        )
        print(f"3. Adjacency Decoder: {adjacency_decoder_name} that decodes {self.adjacency_decoder.in_channels} latent features to {self.adjacency_decoder.out_channels} adjacency features.")

        # Initialize the predictor.
        # Currently, the predictor is hardcoded to be a simple linear layer.
        self.predictor = self._init_predictor(
                            predictor_name=predictor_name,
                            in_channels=self.encoder.dim,
                            out_channels=out_channels,
                        )
        print(f"4. Predictor: Linear layer that transforms {self.encoder.dim} hidden features to {out_channels} dimensional logits.")


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

        # decode the latent node embeddings to recover the adjacency matrix
        h_adj = self.adjacency_decoder(
                        x=h_latent,
                    )

        # forward pass through the predictor
        unnormalized_logits = self.predictor(h_latent)

        return h_latent, \
                xhat, \
                h_adj, \
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
        h_adj_batch, \
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
                        'pred_attr': xhat_batch,
                        'target_attr': train_batch.x,
                        'edge_index': train_batch.edge_index,
                        'batch_size': batch_size,
                        'dispersion': torch.exp(self.dispersion),
                        'h_adj': h_adj_batch,
                        'batch_edge_index': train_batch.edge_index,
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
        h_adj_batch, \
        unnormalized_logits_batch = self(
                                        val_batch.x,
                                        val_batch.edge_index
                                    )

        # prepare dictionary of data required for computing loss
        batch_size = val_batch.batch_size
        val_loss_data = {
                        'pred_attr': xhat_batch,
                        'target_attr': val_batch.x,
                        'edge_index': val_batch.edge_index,
                        'batch_size': batch_size,
                        'dispersion': torch.exp(self.dispersion),
                        'h_adj': h_adj_batch,
                        'batch_edge_index': val_batch.edge_index,
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
        Definition of a single test step of the Vanilla GNN model on the current batch of nodes received from the test dataloader at the current training epoch.

        Parameters
        ----------
        - test_batch: torch_geometric.data.Data
            The input test data (batch of nodes).

        Returns
        -------
        - test_loss: torch.Tensor
            The computed loss for this batch.
        """
        # execute the forward of the Vanilla GNN model
        _, \
        _, \
        _, \
        _ = self(
                test_batch.x,
                test_batch.edge_index
            )

        return torch.tensor(0.0)


    @torch.no_grad()
    def collect_inference_data(
            self,
            dataloader: torch.utils.data.DataLoader
        ) -> dict:
        """
        Collect input data and model outputs for all nodes in the specified dataloader.
        
        Returns
        -------
        - dict
            Dictionary containing inference data with keys: X, edge_index, H_latent, X_hat, H_adj
        """
        X = []
        Y_cell_types = []
        Y_niche_types = []
        XY_coordinates = []
        H_latent = []
        X_hat = []
        H_adj = []
        Logits = []

        for batch in dataloader:
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_types.append(batch.y[:batch_size])
            Y_niche_types.append(batch.y_niche_types[:batch_size])
            XY_coordinates.append(batch.xy_coordinates[:batch_size])

            mask_x = self.set_mask(
                batch_x=batch.x,
                batch_edge_index=batch.edge_index,
                batch_size=batch_size,
                mask_strategy=self.imputation_params['mask_strategy'],
            )

            h_latent, \
            xhat_batch, \
            h_adj, \
            logits = self(
                        mask_x.to(self.device),
                        batch.edge_index.to(self.device)
                    )

            H_latent.append(h_latent[:batch_size])
            X_hat.append(xhat_batch[:batch_size])
            H_adj.append(h_adj[:batch_size])
            Logits.append(logits[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_types = torch.cat(Y_cell_types, dim=0)
        Y_niche_types = torch.cat(Y_niche_types, dim=0)
        XY_coordinates = torch.cat(XY_coordinates, dim=0)
        H_latent = torch.cat(H_latent, dim=0)
        X_hat = torch.cat(X_hat, dim=0)
        H_adj = torch.cat(H_adj, dim=0)
        Logits = torch.cat(Logits, dim=0)

        return {
            'X': X,
            'Y_cell_types': Y_cell_types,
            'Y_niche_types': Y_niche_types,
            'XY_coordinates': XY_coordinates,
            'edge_index': dataloader.data.edge_index,
            'H_latent': H_latent,
            'X_hat': X_hat,
            'H_adj': H_adj,
            'Logits': Logits,
        }