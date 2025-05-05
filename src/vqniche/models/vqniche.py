"""
This file implements the VQNiche Model. It comprises of a VQNiche Encoder, Attribute Decoder, Adjacency Decoder,and a Linear Predictor.

The VQNiche Encoder builds latent node embeddings using a vanilla GNN module (GraphSAGE, GATv2, or GIN) and quantized representations from the latent embeddings using vector quantization module.

The Attribute Decoder is a Linear layer that uses the quantized node embeddings and outputs the predicted attributes. When the attribute reconstruction loss is set to Negative Binomial Loss, the Attribute Decoder is a LinearSoftmax layer which applies a softmax function to the output of the Linear layer and then multiplies the result by the empirical read depth of the target nodes.

The Adjacency Decoder is a Linear layer that uses the quantized node embeddings and outputs the predicted adjacency matrix. The Predictor is a Linear layer that uses the quantized node embeddings and outputs a reconstructed adjacency matrix.

The Predictor builds the logits for the label prediction task using the quantized node embeddings.

The implementation is based on the paper: VQNiche: Rethinking Graph Representation Space for Bridging GNNs and MLPs.
"""
from typing import List, Literal

import networkx as nx

import torch
import torch_geometric

from .base_model import BaseModel
from ..encoders.vqniche_encoder import VQNiche_Encoder
from ..utils import metrics
from ..utils.type_conversions import edge_index_to_adjacency_tensor
from ..utils.metrics import build_reconstructed_adjacency_matrix


class VQNiche(BaseModel):
    def __init__(
            self,
            model_name: Literal['VQNiche'] = 'VQNiche',
            encoder_name: Literal['VQNiche_Encoder'] = 'VQNiche_Encoder',
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            predictor_name: Literal['Linear'] = 'Linear',
            log_similarity_stats: bool = False,
            log_pearson_correlation: bool = False,
            log_mmd_degree: bool = False,
            log_codebook_utilization: bool = True,
            in_channels: int = None,
            out_channels: int = None,
            encoder_params: dict = {},
            attribute_decoder_params: dict = {},
            adjacency_decoder_params: dict = {},
            optimizer_params: dict = {},
            loss_params: dict = {},
        ):
        """
        Initializes the VQNiche model.

        Parameters
        ----------
        - model_name: Literal['VQNiche']
            The name of the model.
        - encoder_name: Literal['VQNiche_Encoder']
            The name of the encoder module.
        - attribute_decoder_name: Literal['MLPSoftmax']
            The name of the attribute decoder module.
        - adjacency_decoder_name: Literal['MLP_AdjacencyDecoder']
            The name of the adjacency decoder module.
        - predictor_name: Literal['Linear']
            The name of the predictor module.
        - log_similarity_stats: bool
            Whether to log the pairwise similarity statistics for all embeddings.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation between original and reconstructed cell-gene matrices.
        - log_mmd_degree: bool
            Whether to log the MMD metric between the degree distribution of the original and reconstructed graphs.
        - log_codebook_utilization: bool
            Whether to log the codebook utilization.

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
        # Initialize the BaseModel class
        super().__init__(
            model_name=model_name,
            encoder_name=encoder_name,
            attribute_decoder_name=attribute_decoder_name,
            adjacency_decoder_name=adjacency_decoder_name,
            predictor_name=predictor_name,
            log_similarity_stats=log_similarity_stats,
            log_codebook_utilization=log_codebook_utilization,
            log_pearson_correlation=log_pearson_correlation,
            log_mmd_degree=log_mmd_degree,
            in_channels=in_channels,
            out_channels=out_channels,
            **optimizer_params,
            **loss_params,
        )

        # Initialize VQNiche encoder module.
        # This module either an MLP module or a GNN module or an MLP followed by a GNN module to build latent node embeddings.
        # Then, it applies a vector quantization module to quantize the latent node embeddings.
        # The out_channels parameter is not passed to the VQNiche_Encoder to separate the encoder from the predictor.
        self.encoder = VQNiche_Encoder(
                            in_channels=in_channels,
                            **encoder_params
                        )

        # Initialize the attribute decoder.
        self.attribute_decoder = self._init_attribute_decoder(
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=attribute_decoder_params,
        )
        print(f"2. Attribute Decoder: {attribute_decoder_name} that decodes {self.encoder.dim} latent features to {in_channels} input features.")

        # Initialize the decoder module for the adjacency matrix
        # Currently, the decoder is hard-coded to be a simple linear layer.
        self.adjacency_decoder = self._init_adjacency_decoder(
            in_channels=self.encoder.dim,
            adjacency_decoder_name=adjacency_decoder_name,
            adjacency_decoder_params=adjacency_decoder_params,
        )
        print(f"3. Adjacency Decoder: {adjacency_decoder_name} that decodes {self.encoder.dim} latent features to {self.encoder.dim} adjacency features.")

        # Initialize the predictor.
        # Currently, the predictor is hardcoded to be a simple linear layer.
        self.predictor = self._init_predictor(
                            predictor_name=predictor_name,
                            in_channels=self.encoder.dim,
                            out_channels=out_channels,
                        )
        print(f"4. Predictor: {predictor_name} that transforms {self.encoder.dim} hidden features to {out_channels} dimensional logits.")


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor,
            batch_xy_coordinates: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the VQNiche model.

        Parameters:
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.
        - batch_xy_coordinates: torch.Tensor
            The spatial coordinates of the batch of nodes.

        Returns
        -------
        - h_latent: torch.Tensor
            Latent node embeddings from the VQNiche Encoder.
        - h_quantized: torch.Tensor
            Quantized node embeddings from the VQNiche Encoder.
        - indices: torch.Tensor
            The indices of the node embeddings mapped to codebook embeddings.
        - xhat: torch.Tensor
            Reconstructed node attributes from the Attribute Decoder.
        - h_edge: torch.Tensor
            The decoded adjacency embeddings from the Adjacency Decoder.
        - unnormalized_logits_batch: torch.Tensor
            Unnormalized logits for the batch of nodes from the Predictor.

        Notes
        -----
        - h_latent is either the output of the MLP module (if no GNN layers are applied) or the output of the GNN module (if no MLP layers are applied) or the output of MLP followed by GNN modules (if both are applied).
        - h_quantized is the output of the VQ module.
        """
        # execute the forward of the VQNiche_Encoder module
        h_latent, \
        h_quantized, \
        indices \
            = self.encoder(
                            batch_x,
                            batch_edge_index
                        )

        if self.attribute_decoder.use_xy_coordinates:
            xhat = self.attribute_decoder(
                        x=torch.cat([h_quantized, batch_xy_coordinates], dim=-1),
                        read_depth=batch_x.sum(dim=-1)
                    )
        else:
            xhat = self.attribute_decoder(
                        x=h_quantized,
                        read_depth=batch_x.sum(dim=-1)
                    )

        # decode the VQ-encoded edge embeddings to recover the adjacency matrix
        h_adj = self.adjacency_decoder(h_quantized)

        unnormalized_logits_batch = self.predictor(h_latent)

        return h_latent, \
            h_quantized, \
            indices, \
            xhat, \
            h_adj, \
            unnormalized_logits_batch


    def training_step(
            self,
            train_batch: torch_geometric.data.Data,
        ) -> torch.Tensor:
        """
        Definition of a single training step of the GraphSAGE model on the current batch of nodes received from the training dataloader at the current training epoch.

        Parameters
        ----------
        - batch: torch_geometric.data.Data
            The input train data (batch of nodes).

        Returns
        -------
        - train_loss: torch.Tensor
            The computed loss for this batch.
        """
        batch_size = train_batch.batch_size

        # execute the forward of the VQNiche model
        h_latent, \
        h_quantized, \
        indices, \
        xhat, \
        h_adj, \
        unnormalized_logits_batch \
            = self(
                    train_batch.x,
                    train_batch.edge_index,
                    train_batch.xy_coordinates,
                )

        # prepare dictionary of data required for computing loss
        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        # print(f"{train_batch.}")
        train_loss_data = {
                        'quantizer_input': h_latent[:batch_size], # code and commit loss
                        'quantizer_output': h_quantized[:batch_size], # code and commit loss
                        'pred_attr': xhat[:batch_size], # attribute reconstruction loss
                        'target_attr': train_batch.x[:batch_size], # attribute reconstruction loss
                        'dispersion': torch.exp(self.dispersion), # attribute reconstruction loss
                        'pred_adj': h_adj[:batch_size], # adjacency reconstruction loss
                        'batch_edge_index': train_batch.edge_index, # adjacency reconstruction loss
                        'batch_input_id': train_batch.input_id, # adjacency reconstruction loss
                        'batch_nid': train_batch.n_id, # adjacency reconstruction loss
                        'logits': unnormalized_logits_batch[:batch_size], # label prediction loss
                        'labels': train_batch.y[:batch_size], # label prediction loss
                        }

        train_loss = self.common_step(
            batch_loss_data=train_loss_data,
            batch_size=batch_size,
            mode='train',
        )

        return train_loss


    def validation_step(
            self,
            val_batch: torch_geometric.data.Data,
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

        h_latent, \
        h_quantized, \
        indices, \
        xhat, \
        h_adj, \
        unnormalized_logits_batch \
            = self(
                    val_batch.x,
                    val_batch.edge_index,
                    val_batch.xy_coordinates,
                )

        # prepare dictionary of data required for computing loss
        val_loss_data = {
                        'quantizer_input': h_latent[:batch_size],
                        'quantizer_output': h_quantized[:batch_size],
                        'pred_attr': xhat[:batch_size],
                        'target_attr': val_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        'pred_adj': h_adj[:batch_size],
                        'batch_edge_index': val_batch.edge_index,
                        'batch_input_id': val_batch.input_id,
                        'batch_nid': val_batch.n_id,
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
        batch_size = test_batch.batch_size

        # execute the forward of the GraphSAGE model
        _, \
        _, \
        _, \
        _, \
        _, \
        unnormalized_logits_batch \
            = self(
                    test_batch.x,
                    test_batch.edge_index,
                    test_batch.xy_coordinates,
                )

        # prepare dictionary of data required for computing accuracy
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
        H_quantized = []
        Indices = []
        X_hat = []
        H_adj = []

        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_type.append(batch.y[:batch_size])
            Y_niche_type.append(batch.y_niche_types[:batch_size])

            h_latent, \
            h_quantized, \
            indices, \
            xhat, \
            h_adj, \
            _ = self(
                    batch.x.to(self.device),
                    batch.edge_index.to(self.device),
                    batch.xy_coordinates.to(self.device)
                )

            H_latent.append(h_latent[:batch_size])
            H_quantized.append(h_quantized[:batch_size])
            Indices.append(indices[:batch_size])
            X_hat.append(xhat[:batch_size])
            H_adj.append(h_adj[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_type = torch.cat(Y_cell_type, dim=0)
        Y_niche_type = torch.cat(Y_niche_type, dim=0)
        H_latent = torch.cat(H_latent, dim=0)
        H_quantized = torch.cat(H_quantized, dim=0)
        Indices = torch.cat(Indices, dim=0)
        X_hat = torch.cat(X_hat, dim=0)
        H_adj = torch.cat(H_adj, dim=0)

        return X, \
            Y_cell_type, \
            Y_niche_type, \
            self.trainer.datamodule.data.edge_index, \
            H_latent, \
            H_quantized, \
            Indices, \
            X_hat, \
            H_adj


    @torch.no_grad()
    def compute_train_epoch_stats(self) -> List[int]:
        """
        If the log_similarity_stats flag is set to True, compute statistics for original and decoded node attributes, and latent and quantized node embeddings at the end of each training epoch.
        If the log_pearson_correlation flag is set to True, compute the Pearson correlation between original and decoded attributes at the end of each training epoch.
        If the log_codebook_utilization flag is set to True, compute the codebook utilization at the end of each training epoch.

        Returns
        -------
        - train_epoch_end_stats: dict
            Dictionary containing the computed statistics.
        """
        X, \
        _, \
        _, \
        edge_index, \
        H_latent, \
        H_quantized, \
        Indices, \
        X_hat, \
        H_adj \
            = self.inference()

        train_epoch_end_stats = {}

        # Concatenate all embeddings
        if self.log_similarity_stats:
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X, 'X')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(H_latent, 'H_latent')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(H_quantized, 'H_quantized')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X_hat, 'X_hat')
            )

        if self.log_pearson_correlation:
            pearson_cell_wise = metrics.pearson_correlation(
                        X.cpu().numpy(),
                        X_hat.cpu().numpy(),
                        compare_genes=False,
                        mean=True,
                    )
            pearson_gene_wise = metrics.pearson_correlation(
                        X.cpu().numpy(),
                        X_hat.cpu().numpy(),
                        compare_genes=True,
                        mean=True,
                    )
            train_epoch_end_stats['pearson_cell_wise'] = pearson_cell_wise
            train_epoch_end_stats['pearson_gene_wise'] = pearson_gene_wise

        if self.log_mmd_degree:
            G = nx.from_numpy_array(
                    edge_index_to_adjacency_tensor(
                        edge_index
                    ).cpu().numpy()
                )

            G_hat = nx.from_numpy_array(
                    build_reconstructed_adjacency_matrix(
                        H_adj
                    ).cpu().numpy()
                )
            node_degree_distribution = metrics.node_degree_distribution(G)
            node_degree_distribution_hat = metrics.node_degree_distribution(G_hat)
            mmd_degree = metrics.mmd_score(
                            [node_degree_distribution],
                            [node_degree_distribution_hat],
                            method='l1_gaussian_tv',
                            sigma=1.0,
                        )
            train_epoch_end_stats['mmd_degree'] = mmd_degree

        if self.log_codebook_utilization:
            train_epoch_end_stats['codebook_utilization'] = 1.0 * len(set(Indices.cpu().numpy())) / self.encoder.vq.codebook.shape[0]
        return train_epoch_end_stats
