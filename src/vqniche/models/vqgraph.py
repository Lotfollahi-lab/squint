"""
Our implementation of VQGraph is built off of the code published by the authors of the VQGraph paper (source: https://github.com/YangLing0818/VQGraph/). We adapt it to work within our setup of Pytorch Geometric, Dataset-Blob, Pytorch Lightning, and Encoder-Predictor setup.

The VQGraph model is a graph neural network model that uses vector quantization (VQ) to encode node embeddings. The VQGraph model consists of a VQGraph_Encoder module and a Linear predictor module. The VQGraph_Encoder module is responsible for encoding the input node embeddings using a graph-convolution module followed by vector quantization, while the Linear predictor module is responsible for predicting the output node embeddings. We use the Euclidean distance as the distance metric for vector quantization (i.e. Euclidean Codebook).
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
            predictor_name: str = 'Linear',
            in_channels: int = None,
            out_channels: int = None,
            apply_vq_on_latent_space: bool = True,
            log_codebook_utilization: bool = False,
            graphconv_layer_name: str = 'SAGEConv',
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
        - log_codebook_utilization: bool
            Whether to log the codebook utilization.

        - graphconv_layer_name: str
            The name of the graph convolutional layer.
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
                            apply_vq_on_latent_space=apply_vq_on_latent_space,
                            graphconv_layer_name=graphconv_layer_name,
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

        self.log_codebook_utilization = log_codebook_utilization


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
        - h_post_vq_conv: torch.Tensor
            Forward (output) of the post-VQ Graph Convolution module.
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
        h_post_vq_conv \
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
            h_post_vq_conv, \
            unnormalized_logits_batch


    # NOTE: Add the following method to the BaseModel class.
    # @torch.no_grad()
    # def embed(


    @torch.no_grad()
    def compute_train_epoch_stats(self) -> List[int]:
        """
        Compute pairwise similarity statistics for all embeddings and codebook utilization.

        Returns
        -------
        - code_indices: List[int]
            List of sorted codebook indices.
        - similarity_stats: dict
            Dictionary containing mean and std of pairwise cosine similarities for different embeddings
        """
        code_indices = []
        h_pre_vq_conv_list = []
        h_post_vq_conv_list = []
        logits_list = []

        # Iterate through inference dataloader
        for batch in self.trainer.datamodule.infer_dataloader():
            batch = batch.to(self.device)
            batch_size = batch.batch_size

            h_pre_vq_conv, _, indices, _, _, _, _, h_post_vq_conv, logits = self(batch.x, batch.edge_index)

            h_pre_vq_conv_list.append(h_pre_vq_conv[:batch_size])
            h_post_vq_conv_list.append(h_post_vq_conv[:batch_size])
            logits_list.append(logits[:batch_size])
            code_indices.extend(indices[:batch_size].tolist())

        # Concatenate all embeddings
        h_pre_vq_conv = torch.cat(h_pre_vq_conv_list, dim=0)
        h_post_vq_conv = torch.cat(h_post_vq_conv_list, dim=0)
        logits = torch.cat(logits_list, dim=0)

        # Compute statistics for all embeddings
        similarity_stats = {}
        similarity_stats.update(
            metrics.get_similarity_stats(h_pre_vq_conv, 'h_pre_vq_conv')
        )
        similarity_stats.update(
            metrics.get_similarity_stats(h_post_vq_conv, 'h_post_vq_conv')
        )
        similarity_stats.update(
            metrics.get_similarity_stats(logits, 'logits')
        )
        similarity_stats.update(
            metrics.get_similarity_stats(self.encoder.codebook, 'codebook')
        )

        return code_indices, similarity_stats


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
        dist, \
        codebook_embeddings, \
        h_node, \
        h_edge, \
        h_post_vq_conv, \
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
                        'target_attr': torch.log1p(train_batch.x[:batch_size]),
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
            dist, \
            codebook_embeddings, \
            h_node, \
            h_edge, \
            h_post_vq_conv, \
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
                        'target_attr': torch.log1p(val_batch.x[:batch_size]),
                        'pred_adj': h_edge[:batch_size],
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
        if self.log_codebook_utilization:
            code_indices, similarity_stats = self.compute_train_epoch_stats()
            self.log(
                name="codebook_utilization",
                value=len(set(code_indices)),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            for key, value in similarity_stats.items():
                self.log(
                    name=key,
                    value=value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return super().on_train_epoch_end()
