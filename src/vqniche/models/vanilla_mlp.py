"""
This file implements a Vanilla MLP Model.

The Vanilla MLP Model comprises of the following components:
- Encoder: A MLP module.
- Attribute Decoder: A MLPSoftmax decoder.
- Predictor: A Linear predictor.

In addition, this provides the option to log the mean pairwise cosine similarity between the original attributes, decoded attributes, and MLP embeddings, and the Pearson correlation between the original and decoded node attributes at the end of each training epoch.
"""
from typing import Literal

import torch
import torch_geometric

from .base_model import BaseModel
from ..modules.mlp import MLP as MLP_Encoder


class VanillaMLP(BaseModel):
    def __init__(
            self,
            model_name: Literal['VanillaMLP'] = 'VanillaMLP',
            encoder_name: Literal['MLP_Encoder'] = 'MLP_Encoder',
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            predictor_name: Literal['Linear'] = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            train_log_flags: dict = {},
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
        - model_name: Literal['VanillaMLP']
            The name of the model.
        - encoder_name: Literal['MLP_Encoder']
            The name of the encoder module.
        - attribute_decoder_name: Literal['MLPSoftmax']
            The name of the attribute decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - predictor_name: Literal['Linear']
            The name of the predictor module.

        - in_channels: int
            The number of input features.
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
        # Initialize the BaseModel class
        super().__init__(
            model_name=model_name,
            encoder_name=encoder_name,
            attribute_decoder_name=attribute_decoder_name,
            adjacency_decoder_name=adjacency_decoder_name,
            predictor_name=predictor_name,
            in_channels=in_channels,
            out_channels=out_channels,
            **train_log_flags,
            **optimizer_params,
            **loss_params,
        )

        # Initialize an MLP module as the encoder.
        self.encoder = MLP_Encoder(
            in_channels=in_channels,
            mlp_params=encoder_params['mlp_params'],
        )
        print(f"1. MLP Encoder: {self.encoder.num_layers} Linear layer(s) that transform {self.encoder.channel_list[0]} input features to {self.encoder.channel_list[-1]} latent features.")

        # Initialize the attribute decoder.
        self.attribute_decoder = self._init_attribute_decoder(
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=attribute_decoder_params,
        )
        print(f"2. Attribute Decoder: {attribute_decoder_name} that decodes {self.encoder.channel_list[-1]} latent features to {in_channels} input features.")

        # Initialize the adjacency decoder.
        self.adjacency_decoder = self._init_adjacency_decoder(
            in_channels=self.encoder.channel_list[-1],
            adjacency_decoder_name=adjacency_decoder_name,
            adjacency_decoder_params=adjacency_decoder_params,
        )
        print(f"3. Adjacency Decoder: {adjacency_decoder_name} that decodes {self.encoder.channel_list[-1]} latent features to {self.encoder.channel_list[-1]} adjacency features.")

        # Initialize the predictor.
        # Currently, the predictor is hardcoded to be a simple linear layer.
        self.predictor = self._init_predictor(
            predictor_name=predictor_name,
            in_channels=self.encoder.channel_list[-1],
            out_channels=out_channels,
        )
        print(f"4. Predictor: Linear layer that transforms {self.encoder.channel_list[-1]} hidden features to {out_channels} dimensional logits.")


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
        - h_latent: torch.Tensor
            Latent node embeddings from the MLP Encoder.
        - xhat: torch.Tensor
            Reconstructed node attributes from the Attribute Decoder.
        - unnormalized_logits: torch.Tensor
            Unnormalized logits from the Predictor.
        """
        # forward pass through the MLP encoder
        h_latent = self.encoder(
                        batch_x,
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
        unnormalized_logits_batch = self(train_batch.x)

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
        unnormalized_logits_batch = self(val_batch.x)

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
        Y_cell_type = []
        Y_niche_type = []
        XY_coordinates = []
        H_latent = []
        X_hat = []
        H_adj = []

        for batch in dataloader:
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_type.append(batch.y[:batch_size])
            Y_niche_type.append(batch.y_niche_types[:batch_size])
            XY_coordinates.append(batch.xy_coordinates[:batch_size])

            h_latent, \
            xhat_batch, \
            h_adj_batch, \
            _ = self(batch.x.to(self.device))

            H_latent.append(h_latent[:batch_size])
            X_hat.append(xhat_batch[:batch_size])
            H_adj.append(h_adj_batch[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_type = torch.cat(Y_cell_type, dim=0)
        Y_niche_type = torch.cat(Y_niche_type, dim=0)
        XY_coordinates = torch.cat(XY_coordinates, dim=0)
        H_latent = torch.cat(H_latent, dim=0)
        X_hat = torch.cat(X_hat, dim=0)
        H_adj = torch.cat(H_adj, dim=0)

        return {
            'X': X,
            'Y_cell_type': Y_cell_type,
            'Y_niche_type': Y_niche_type,
            'XY_coordinates': XY_coordinates,
            'edge_index': dataloader.data.edge_index,
            'H_latent': H_latent,
            'X_hat': X_hat,
            'H_adj': H_adj
        }