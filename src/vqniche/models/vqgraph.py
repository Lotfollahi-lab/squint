"""
This file implements the VQGraph Model. It comprises of an Encoder and a Linear Predictor. The Encoder jointly trains a node embedding using a graph-convolution module and a codebook using a vector quantization module. The Predictor is a Linear layer that uses the quantized node embeddings and outputs the predicted labels.

The implementation is based on the paper: VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs.
"""
from typing import List, Union, Callable, Literal

import torch
import torch.nn as nn
import torch_geometric

from .base_model import BaseModel
from ..encoders.vqgraph_encoder import VQGraph_Encoder
from ..utils import metrics


class VQGraph(BaseModel):
    def __init__(
            self,
            model_name: str = 'VQGraph',
            encoder_name: str = 'VQGraph_Encoder',
            attribute_decoder_name: Literal['Linear', 'LinearSoftmax'] = 'Linear',
            predictor_name: str = 'Linear',
            log_similarity_stats: bool = False,
            log_pearson_correlation: bool = False,
            log_codebook_utilization: bool = True,
            in_channels: int = None,
            out_channels: int = None,
            gnn_layer_name: str = 'SAGEConv',
            hidden_channels: int = 64,
            num_layers: int = 2,
            act_first: bool = True,
            activation: Union[str, Callable, None] = "relu",
            norm: Union[str, Callable, None] = None,
            dropout: float = 0.5,
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
            codebook_params: dict = {},
            optimizer_name: str = 'adam',
            lr: float = 0.01,
            weight_decay: float = 0.0,
            loss_names: List[str] = ['cross_entropy'],
            loss_kwargs: dict = {'reduction': 'none'},
        ):
        """
        Initializes the VQGraph model.

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
            Whether to log the pairwise similarity statistics for all embeddings.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation between original and reconstructed cell-gene matrices.
        - log_codebook_utilization: bool
            Whether to log the codebook utilization.

        - in_channels: int
            The number of input features.
        - out_channels: int
            The number of output features.

        - gnn_layer_name: str
            The name of the GNN layer.
        - attribute_decoder_name: Literal['Linear', 'LinearSoftmax']
            The name of the attribute decoder module.
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
        - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
            The initialization method to use for the linear transformations in the SAGEConv layers.
            If None, the initialization method is 'kaiming_uniform'.
        - codebook_params: dict
            Keyword arguments for the codebook.

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
            log_codebook_utilization=log_codebook_utilization,
            log_pearson_correlation=log_pearson_correlation,
            in_channels=in_channels,
            out_channels=out_channels,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            loss_names=loss_names,
            loss_kwargs=loss_kwargs,
        )

        # Initialize VQGraph encoder module.
        # The out_channels parameter is not passed to the VQGraph_Encoder (i.e. it is set to None) so that we can separate the encoder from the predictor.
        self.encoder = VQGraph_Encoder(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            gnn_layer_name=gnn_layer_name,
                            num_layers=num_layers,
                            act_first=act_first,
                            activation=activation,
                            dropout=dropout,
                            norm=norm,
                            init_method=init_method,
                            **codebook_params
                        )

        # Initialize the attribute decoder.
        print(f"Initializing the {attribute_decoder_name} attribute decoder module that takes as input latent node embeddings of dimension {hidden_channels} and outputs an estimate of {in_channels} original features.")
        self.attribute_decoder = self._init_attribute_decoder(
                                attribute_decoder_name=attribute_decoder_name,
                                in_channels=hidden_channels,
                                out_channels=in_channels
                            )

        # Initialize the decoder module for the adjacency matrix
        # Currently, the decoder is hard-coded to be a simple linear layer.
        print(f"Initializing the decoder module for the adjacency matrix with {hidden_channels} input dimension and {hidden_channels} output dimension.")
        self.decoder_edge = nn.Linear(
                                in_features=hidden_channels,
                                out_features=hidden_channels
                            )

        # Instead, we apply this final linear transformation in the predictor module manually to have access to the internal node embeddings via the `embed` function.
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
        Forward pass of the VQGraph model.

        Parameters:
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.

        Returns
        -------
        - h_pre_vq_conv: torch.Tensor
            Forward (output) of the pre-VQ Graph Convolution module.
        - h_vq: torch.Tensor
            VQ-encoded node embeddings.
        - indices: torch.Tensor
            The indices of the node embeddings mapped to codebook embeddings.
        - dist: torch.Tensor
            The distances between the node embeddings and the codebook embeddings.
        - codebook_embeddings: torch.Tensor
            The codebook embeddings.
        - h_node: torch.Tensor
            The decoded node attributes.
        - h_edge: torch.Tensor
            The decoded adjacency embeddings.
        - unnormalized_logits_batch: torch.Tensor
            The unnormalized logits for the batch of nodes (output of the predictor module).
        """
        # execute the forward of the VQGraph_Encoder model
        h_gnn, \
        h_vq, \
        indices, \
        dist, \
        codebook_embeddings, \
            = self.encoder(
                            batch_x,
                            batch_edge_index
                        )

        h_node = self.attribute_decoder(
                    x=h_vq,
                    read_depth=batch_x.sum(dim=-1)
                )

        # decode the VQ-encoded edge embeddings to recover the adjacency matrix
        h_edge = self.decoder_edge(h_vq)

        unnormalized_logits_batch = self.predictor(h_vq)

        return h_gnn, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings, \
            h_node, \
            h_edge, \
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

        # execute the forward of the VQGraph model
        h_gnn, \
        h_vq, \
        indices, \
        _, \
        codebook_embeddings, \
        h_node, \
        h_edge, \
        unnormalized_logits_batch \
            = self(
                    train_batch.x,
                    train_batch.edge_index,
                )

        # prepare dictionary of data required for computing loss
        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        train_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': train_batch.y[:batch_size],
                        'pred_attr': h_node[:batch_size],
                        'target_attr': train_batch.x[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        'pred_adj': h_edge[:batch_size],
                        'batch_edge_index': train_batch.edge_index,
                        'quantizer_input': h_gnn[:batch_size],
                        'quantized_output': h_vq[:batch_size],
                        'node_embeddings': h_gnn[:batch_size],
                        'codebook_embeddings': codebook_embeddings[0],
                        'code_indices': indices[:batch_size],
                        'batch_input_id': train_batch.input_id,
                        'batch_nid': train_batch.n_id,
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

        h_gnn, \
        h_vq, \
        indices, \
        _, \
        codebook_embeddings, \
        h_node, \
        h_edge, \
        unnormalized_logits_batch \
            = self(
                    val_batch.x,
                    val_batch.edge_index,
                )

        # prepare dictionary of data required for computing loss
        val_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
                        'pred_attr': h_node[:batch_size],
                        'target_attr': val_batch.x[:batch_size],
                        'pred_adj': h_edge[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        'batch_edge_index': val_batch.edge_index,
                        'quantizer_input': h_gnn[:batch_size],
                        'quantized_output': h_vq[:batch_size],
                        'node_embeddings': h_gnn[:batch_size],
                        'codebook_embeddings': codebook_embeddings[0],
                        'code_indices': indices[:batch_size],
                        'batch_input_id': val_batch.input_id,
                        'batch_nid': val_batch.n_id,
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
        _, \
        _, \
        unnormalized_logits_batch \
            = self(
                    test_batch.x,
                    test_batch.edge_index
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
        H_gnn = []
        H_vq = []
        Indices = []
        X_hat = []
        H_edge = []

        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_type.append(batch.y[:batch_size])
            Y_niche_type.append(batch.y_niche_types[:batch_size])

            h_gnn, \
            h_vq, \
            indices, \
            _, \
            _, \
            h_node, \
            h_edge, \
            _ = self(
                    batch.x.to(self.device),
                    batch.edge_index.to(self.device)
                )

            H_gnn.append(h_gnn[:batch_size])
            H_vq.append(h_vq[:batch_size])
            Indices.append(indices[:batch_size])
            X_hat.append(h_node[:batch_size])
            H_edge.append(h_edge[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_type = torch.cat(Y_cell_type, dim=0)
        Y_niche_type = torch.cat(Y_niche_type, dim=0)
        H_gnn = torch.cat(H_gnn, dim=0)
        H_vq = torch.cat(H_vq, dim=0)
        Indices = torch.cat(Indices, dim=0)
        X_hat = torch.cat(X_hat, dim=0)
        H_edge = torch.cat(H_edge, dim=0)

        return X, \
            Y_cell_type, \
            Y_niche_type, \
            H_gnn, \
            H_vq, \
            Indices, \
            X_hat, \
            H_edge


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
        H_gnn, \
        H_vq, \
        Indices, \
        X_hat, \
        H_edge = self.inference()

        train_epoch_end_stats = {}

        # Concatenate all embeddings
        if self.log_similarity_stats:
            train_epoch_end_stats.update(
                metrics.cosine_similarity(X, 'X')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(H_gnn, 'H_gnn')
            )
            train_epoch_end_stats.update(
                metrics.cosine_similarity(H_vq, 'H_vq')
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

        if self.log_codebook_utilization:
            train_epoch_end_stats['codebook_utilization'] = 1.0 * len(set(Indices.cpu().numpy())) / self.encoder.codebook.shape[0]

        return train_epoch_end_stats
