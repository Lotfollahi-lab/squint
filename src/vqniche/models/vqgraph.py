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
from ..utils.metrics import compute_pearson_correlation, compute_mmd

class VQGraph(BaseModel):
    def __init__(
            self,
            model_name: str = 'VQGraph',
            encoder_name: str = 'VQGraph_Encoder',
            predictor_name: str = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            log_similarity_stats: bool = False,
            log_codebook_utilization: bool = True,
            log_pearson_correlation: bool = False,
            graphconv_layer_name: str = 'SAGEConv',
            attribute_decoder_name: Literal['Linear', 'LinearSoftmax'] = 'Linear',
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
            inference_mode: Literal['batch-wise', 'layer-wise'] = 'batch-wise',
        ):
        """
        Initializes the VQGraph model.

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
        - apply_vq_on_latent_space: bool
            Whether to apply vector quantization on the latent space or the input space.
        - log_similarity_stats: bool
            Whether to log the pairwise similarity statistics for all embeddings.
        - log_codebook_utilization: bool
            Whether to log the codebook utilization.
        - log_pearson_correlation: bool
            Whether to log the Pearson correlation between original and reconstructed cell-gene matrices.

        - graphconv_layer_name: str
            The name of the graph convolutional layer.
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

        - inference_mode: str
            The inference mode. Choose from 'batch-wise' or 'layer-wise'.
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
            inference_mode=inference_mode,
        )

        # Initialize VQGraph encoder module.
        # The out_channels parameter is not passed to the VQGraph_Encoder (i.e. it is set to None) so that we can separate the encoder from the predictor.
        self.encoder = VQGraph_Encoder(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            graphconv_layer_name=graphconv_layer_name,
                            attribute_decoder_name=attribute_decoder_name,
                            num_layers=num_layers,
                            act_first=act_first,
                            activation=activation,
                            dropout=dropout,
                            norm=norm,
                            init_method=init_method,
                            **codebook_params
                        )

        # Instead, we apply this final linear transformation in the predictor module manually to have access to the internal node embeddings via the `embed` function.
        self.predictor = nn.Linear(
                            in_features=hidden_channels,
                            out_features=out_channels
                        )

        self.log_similarity_stats = log_similarity_stats
        self.log_codebook_utilization = log_codebook_utilization
        self.log_pearson_correlation = log_pearson_correlation


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
        h_pre_vq_conv, \
        h_vq, \
        indices, \
        dist, \
        codebook_embeddings, \
        h_node, \
        h_edge, \
            = self.encoder(
                            batch_x,
                            batch_edge_index
                        )

        unnormalized_logits_batch = self.predictor(h_vq)

        return h_pre_vq_conv, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings, \
            h_node, \
            h_edge, \
            unnormalized_logits_batch


    @torch.no_grad()
    def inference(self):
        X = []
        H_pre_vq_conv = []
        H_vq = []
        Indices = []
        X_hat = []
        H_edge = []
        Labels_cell_type = []
        Labels_niche_type = []

        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size
            Labels_cell_type.append(batch.y[:batch_size])
            Labels_niche_type.append(batch.y_niche_types[:batch_size])

            h_pre_vq_conv, \
            h_vq, \
            indices, \
            _, \
            _, \
            h_node, \
            h_edge, \
            _ = self(batch.x, batch.edge_index)

            X.append(batch.x[:batch_size])
            H_pre_vq_conv.append(h_pre_vq_conv[:batch_size])
            H_vq.append(h_vq[:batch_size])
            Indices.append(indices[:batch_size])
            X_hat.append(h_node[:batch_size])
            H_edge.append(h_edge[:batch_size])

        X = torch.cat(X, dim=0)
        H_pre_vq_conv = torch.cat(H_pre_vq_conv, dim=0)
        H_vq = torch.cat(H_vq, dim=0)
        Indices = torch.cat(Indices, dim=0)
        X_hat = torch.cat(X_hat, dim=0)
        H_edge = torch.cat(H_edge, dim=0)
        Labels_cell_type = torch.cat(Labels_cell_type, dim=0)
        Labels_niche_type = torch.cat(Labels_niche_type, dim=0)

        return X, \
            Labels_cell_type, \
            Labels_niche_type, \
            H_pre_vq_conv, \
            H_vq, \
            Indices, \
            X_hat, \
            H_edge


    @torch.no_grad()
    def compute_train_epoch_stats(self) -> List[int]:
        """
        Compute train epoch end statistics: pairwise similarity statistics for all embeddings, codebook utilization, Pearson correlation between original and reconstructed cell-gene matrices, and MMD for node degree distribution.

        Returns
        -------
        - train_epoch_end_stats: dict
            Dictionary containing pairwise similarity statistics for all embeddings, codebook utilization, Pearson correlation between original and reconstructed cell-gene matrices, and MMD for node degree distribution.
        """
        train_epoch_end_stats = {}

        if self.log_similarity_stats:
            h_pre_vq_conv_list = []
            logits_list = []
        if self.log_codebook_utilization:
            code_indices = []
        if self.log_pearson_correlation:
            X = []
            X_hat = []

        # Iterate through inference dataloader
        for batch in self.trainer.datamodule.infer_dataloader():
            batch_size = batch.batch_size
            h_pre_vq_conv, \
            _, \
            indices, \
            _, \
            _, \
            h_node, \
            h_edge, \
            logits \
                = self(
                    batch.x.to(self.device),
                    batch.edge_index.to(self.device)
                )

            if self.log_similarity_stats:
                h_pre_vq_conv_list.append(h_pre_vq_conv[:batch_size])
                logits_list.append(logits[:batch_size])

            if self.log_codebook_utilization:
                code_indices.extend(indices[:batch_size].tolist())

            if self.log_pearson_correlation:
                X.append(batch.x[:batch_size])
                X_hat.append(h_node[:batch_size])

        # Concatenate all embeddings
        if self.log_similarity_stats:
            h_pre_vq_conv = torch.cat(h_pre_vq_conv_list, dim=0)
            logits = torch.cat(logits_list, dim=0)

            # Compute statistics for all embeddings
            train_epoch_end_stats.update(
                metrics.get_similarity_stats(h_pre_vq_conv, 'h_pre_vq_conv')
            )
            train_epoch_end_stats.update(
                metrics.get_similarity_stats(logits, 'logits')
            )
            train_epoch_end_stats.update(
                metrics.get_similarity_stats(self.encoder.codebook, 'codebook')
            )

        if self.log_codebook_utilization:
            codebook_utilization = 1.0 * len(set(code_indices)) / self.encoder.codebook.shape[0]
            train_epoch_end_stats['codebook_utilization'] = codebook_utilization

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
        h_pre_vq_conv, \
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
                        'quantizer_input': h_pre_vq_conv[:batch_size],
                        'quantized_output': h_vq[:batch_size],
                        'node_embeddings': h_pre_vq_conv[:batch_size],
                        'codebook_embeddings': codebook_embeddings[0],
                        'code_indices': indices[:batch_size],
                        'batch_input_id': train_batch.input_id,
                        'batch_nid': train_batch.n_id,
                        }

        # compute train loss
        train_loss = self.criterion(
                        loss_data=train_loss_data,
                        curr_batch_size=batch_size,
                        mode='train',
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
            val_batch: torch_geometric.data.Data,
            batch_idx: int
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

        if self.inference_mode == 'batch-wise':
            h_pre_vq_conv, \
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

        elif self.inference_mode == 'layer-wise':
            raise NotImplementedError("Layer-wise inference is not supported for validation step.")

        # prepare dictionary of data required for computing loss
        val_loss_data = {
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
                        'pred_attr': h_node[:batch_size],
                        'target_attr': val_batch.x[:batch_size],
                        'pred_adj': h_edge[:batch_size],
                        'dispersion': torch.exp(self.dispersion),
                        'batch_edge_index': val_batch.edge_index,
                        'quantizer_input': h_pre_vq_conv[:batch_size],
                        'quantized_output': h_vq[:batch_size],
                        'node_embeddings': h_pre_vq_conv[:batch_size],
                        'codebook_embeddings': codebook_embeddings[0],
                        'code_indices': indices[:batch_size],
                        'batch_input_id': val_batch.input_id,
                        'batch_nid': val_batch.n_id,
                        }

        # compute validation loss
        val_loss = self.criterion(
                        loss_data=val_loss_data,
                        curr_batch_size=batch_size,
                        mode='val',
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

        if self.inference_mode == 'batch-wise':
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

        elif self.inference_mode == 'layer-wise':
            raise NotImplementedError("Layer-wise inference is not supported for test step.")

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


    def on_train_epoch_end(self) -> None:
        train_epoch_end_stats = self.compute_train_epoch_stats()
        for key, value in train_epoch_end_stats.items():
            self.log(
                name=key,
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                    sync_dist=True,
                )
        return super().on_train_epoch_end()
