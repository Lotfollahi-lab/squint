"""
This file implements the VQNiche Model. It comprises of a VQNiche Encoder, Attribute Decoder, Adjacency Decoder,and a Linear Predictor.

The VQNiche Encoder builds latent node embeddings using a vanilla GNN module (GraphSAGE, GATv2, or GIN) and quantized representations from the latent embeddings using vector quantization module.

The Attribute Decoder is a Linear layer that uses the quantized node embeddings and outputs the predicted attributes. When the attribute reconstruction loss is set to Negative Binomial Loss, the Attribute Decoder is a LinearSoftmax layer which applies a softmax function to the output of the Linear layer and then multiplies the result by the empirical read depth of the target nodes.

The Adjacency Decoder is a Linear layer that uses the quantized node embeddings and outputs the predicted adjacency matrix. The Predictor is a Linear layer that uses the quantized node embeddings and outputs a reconstructed adjacency matrix.

The Predictor builds the logits for the label prediction task using the quantized node embeddings.

The implementation is based on the paper: VQNiche: Rethinking Graph Representation Space for Bridging GNNs and MLPs.
"""
from typing import Literal, List, Optional

import torch
import torch_geometric

from .base_model import BaseModel
from ..encoders.vqniche_encoder import VQNiche_Encoder
from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features
from vqniche.utils.mask import (
    set_mask_ratio,
    set_mask_indices,
    print_masked_input_diversity_stats,
)


class VQNiche(BaseModel):
    def __init__(
            self,
            model_name: Literal['VQNiche'] = 'VQNiche',
            encoder_name: Literal['VQNiche_Encoder'] = 'VQNiche_Encoder',
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            predictor_name: Literal['Linear'] = 'Linear',
            train_metrics_list: List[str] = [],
            in_channels: int = None,
            out_channels: int = None,
            imputation_params: dict = {},
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
        - imputation_params: dict
            The parameters for the imputation module.
        - encoder_name: Literal['VQNiche_Encoder']
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

        - train_metrics_list: List[str]
            The list of metrics to compute during training.

        - imputation_params: dict
            The parameters for the imputation module.
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
            in_channels=in_channels,
            out_channels=out_channels,
            train_metrics_list=train_metrics_list,
            **optimizer_params,
            **loss_params,
        )

        # Initialize imputation parameters
        print(f"Setting imputation parameters:")
        for key, value in imputation_params.items():
            print(f"{key}: {value}")
        self.mask_strategy = imputation_params['mask_strategy']
        self.base_mask_ratio = imputation_params['base_mask_ratio']
        self.final_mask_ratio = imputation_params['final_mask_ratio']
        self.warmup_epochs = imputation_params['warmup_epochs']
        self.deterministic_masking = imputation_params['deterministic_masking']
        self.compute_mask_input_diversity = imputation_params['compute_mask_input_diversity']
        self.mask_token_eps = imputation_params['mask_token_eps']

        if self.mask_strategy == 'zeros':
            self.mask_token = torch.zeros(self.in_channels)
        elif self.mask_strategy == 'learnable_parameter':
            self.mask_token = torch.nn.Parameter(torch.empty(self.in_channels))
            torch.nn.init.normal_(self.mask_token, mean=10.0,std=1.0)

        # Initialize VQNiche encoder module.
        # This module either an MLP module or a GNN module or an MLP followed by a GNN module to build latent node embeddings.
        # Then, it applies a vector quantization module to quantize the latent node embeddings.
        # The out_channels parameter is not passed to the VQNiche_Encoder to separate the encoder from the predictor.
        self.encoder = VQNiche_Encoder(
                            in_channels=in_channels,
                            **encoder_params
                        )
        print(f"1. VQNiche Encoder: {encoder_name} that transforms {in_channels} input features to {self.encoder.dim} quantized features.")

        # Initialize the attribute decoder.
        # the input dimension of the attribute decoder is the dimension of the quantized node embeddings.
        # the output dimension of the attribute decoder is the dimension of the input features.
        self.attribute_decoder = self._init_attribute_decoder(
            in_channels=self.encoder.dim,
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=attribute_decoder_params,
        )
        print(f"2. Attribute Decoder: {attribute_decoder_name} that decodes quantized latent embeddings of dimension {self.encoder.dim} to input features of dimension {in_channels}.")

        # Initialize the decoder module for the adjacency matrix
        # the 
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
        print(f"4. Predictor: {predictor_name} that transforms {self.encoder.dim} hidden features to {out_channels} dimensional logits.")


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor,
            batch_encoder_conditions: Optional[torch.Tensor] = None,
            batch_attr_decoder_conditions: Optional[torch.Tensor] = None,
            batch_adj_decoder_conditions: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Forward pass of the VQNiche model.

        Parameters:
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.
        - batch_encoder_conditions: torch.Tensor
            The conditioning features for the encoder of the batch of nodes.
        - batch_attr_decoder_conditions: torch.Tensor
            The conditioning features for the attribute decoder of the batch of nodes.
        - batch_adj_decoder_conditions: torch.Tensor
            The conditioning features for the adjacency decoder of the batch of nodes.

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
                            batch_edge_index,
                            batch_encoder_conditions,
                        )

        xhat = self.attribute_decoder(
                    x=h_quantized,
                    read_depth=batch_x.sum(dim=-1),
                    conditions=batch_attr_decoder_conditions,
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



    def prepare_masked_input(
            self,
            batch_x: torch.Tensor,
            mask_idx: torch.BoolTensor,
        ) -> torch.Tensor:
        """
        Mask the input features based on the mask indices.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - mask_idx: torch.BoolTensor
            The indices of the nodes to mask.

        Returns
        -------
        - mask_x: torch.Tensor
            The masked input features.
        """
        # clone the input features to avoid modifying the original features
        masked_x = batch_x.clone()
        
        # if the mask strategy is zeros or learnable_parameter, mask the input features of the source nodes where mask_idx is True with the mask token
        if self.mask_strategy in ['zeros', 'learnable_parameter']:
            # nonzero() returns indices of the `True` elements in the mask_idx tensor
            # squeeze(1) ensures that the index tensor is 1D
            index = mask_idx.to(masked_x.device).nonzero().squeeze(1)
            
            # during training, add a small jitter to the mask token
            if self.training:
                mask = self.mask_token + torch.randn_like(self.mask_token) * self.mask_token_eps
            else:
                mask = self.mask_token
            mask = mask.to(dtype=masked_x.dtype, device=masked_x.device)
            
            # unsqueeze(0) adds a second dimension to the mask token tensor
            # numel() returns the number of elements in the index tensor
            # i.e. numel() is the number of source nodes to mask
            # expand() replicates the mask token tensor to match the number of source nodes to mask
            mask = mask.unsqueeze(0).expand(index.numel(), -1)

            # index_copy_() copies the mask tensor to the masked_x tensor at the specified indices
            # index[i] == j => the i-th row of mask is copied to the j-th row (dim=0) of masked_x
            masked_x.index_copy_(
                dim=0,
                index=index,
                source=mask,
            )
            
        return masked_x


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
        
        # --------------------- Prepare Inputs for VQNiche's Forward Pass ---------------------
        # 1) Set ratio of source nodes to mask
        if self.mask_strategy in ['zeros', 'learnable_parameter']:
            # linearly increase mask ratio across epochs
            mask_ratio: float = set_mask_ratio(
                            epoch=self.current_epoch,
                            base_ratio=self.base_mask_ratio,
                            final_ratio=self.final_mask_ratio,
                            warmup_epochs=self.warmup_epochs,
                        )
        elif self.mask_strategy == 'original':
            # do not mask any nodes
            mask_ratio = 0.0
        
        # 2) Choose which source nodes to mask
        mask_idx: torch.BoolTensor = set_mask_indices(
            N=train_batch.x.size(0),
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            deterministic=self.deterministic_masking
        )

        # 3) Set the mask over the input features based on the mask strategy and mask indices
        masked_x = self.prepare_masked_input(
            batch_x=train_batch.x,
            mask_idx=mask_idx,
        )

        if self.compute_mask_input_diversity:
            print_masked_input_diversity_stats(
                x_in=masked_x,
                batch_size=train_batch.batch_size,
                mask_idx=mask_idx,
            )

        # 3) Prepare conditioning features for the encoder, attribute decoder, and adjacency decoder
        train_encoder_conditions = getattr(train_batch, 'encoder_conditions', None)
        train_attr_decoder_conditions = getattr(train_batch, 'attr_decoder_conditions', None)
        train_adj_decoder_conditions = getattr(train_batch, 'adj_decoder_conditions', None)

        # --------------------- Execute VQNiche's Forward Pass ---------------------
        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch \
            = self(
                batch_x=masked_x,
                batch_edge_index=train_batch.edge_index,
                batch_encoder_conditions=train_encoder_conditions,
                batch_attr_decoder_conditions=train_attr_decoder_conditions,
                batch_adj_decoder_conditions=train_adj_decoder_conditions,
            )

        # --------------------- Prepare Data for Loss Computation ---------------------
        # 1) If k_hop_nb_loss is 1, compute loss over aggregate neighbor features
        if self.loss_kwargs['k_hop_nb_loss'] == 1:
            pred_attr = aggregate_1hop_neighbor_features(
                X=xhat_batch,
                edge_index=train_batch.edge_index,
                return_mean=True,
            )
            target_attr = aggregate_1hop_neighbor_features(
                X=train_batch.x,
                edge_index=train_batch.edge_index,
                return_mean=True,
            )
        # 2) If k_hop_nb_loss is 0, compute loss over individual node features
        elif self.loss_kwargs['k_hop_nb_loss'] == 0:
            pred_attr = xhat_batch
            target_attr = train_batch.x
        else:
            raise ValueError(f"Invalid k_hop_nb_loss: {self.loss_kwargs['k_hop_nb_loss']}")
        
        # 3) If only_masked is True, compute loss over masked nodes only
        if self.loss_kwargs['only_masked']:
            # if only_masked is True and there are masked nodes, compute loss over masked nodes only
            if mask_idx.sum() > 0:
                pred_attr = pred_attr[mask_idx==1]
                target_attr = target_attr[mask_idx==1]
            # if only_masked is True and there are no masked nodes, compute loss over all nodes
            # this is to handle the case where the mask ratio is 0 in a given epochfor zeros and learnable_parameter mask strategies
            else:
                pred_attr = pred_attr[:batch_size]
                target_attr = target_attr[:batch_size]
        # 4) If only_masked is False, compute loss over all nodes
        elif not self.loss_kwargs['only_masked']:
            pred_attr = pred_attr[:batch_size]
            target_attr = target_attr[:batch_size]

        # print(f"{pred_attr.shape=} | {target_attr.shape=}")

        # prepare dictionary of data required for computing loss
        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        train_loss_data = {
                        'quantizer_input': h_latent[:batch_size], # code and commit loss
                        'quantizer_output': h_quantized[:batch_size], # code and commit loss
                        'pred_attr': pred_attr, # attribute reconstruction loss
                        'target_attr': target_attr, # attribute reconstruction loss
                        'edge_index': train_batch.edge_index, # attribute reconstruction loss
                        'batch_size': batch_size, # attribute and adjacency reconstruction loss
                        'dispersion': torch.exp(self.dispersion), # attribute reconstruction loss
                        'h_adj': h_adj_batch, # adjacency reconstruction loss
                        'batch_edge_index': train_batch.edge_index, # adjacency reconstruction loss
                        'logits': unnormalized_logits_batch[:batch_size], # label prediction loss
                        'labels': train_batch.y[:batch_size], # label prediction loss
                        }

        # --------------------- Compute Loss ---------------------
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
        # execute the forward of the VQNiche model
        val_encoder_conditions = getattr(val_batch, 'encoder_conditions', None)
        val_attr_decoder_conditions = getattr(val_batch, 'attr_decoder_conditions', None)
        val_adj_decoder_conditions = getattr(val_batch, 'adj_decoder_conditions', None)

        # option 3: set the input features of the val nodes as mean of the input features of its neighbors using the edge_index
        # Initialize output tensor with zeros for validation nodes and original features for remaining nodes
        mask_x = torch.zeros_like(val_batch.x)
        mask_x[val_batch.batch_size:] = val_batch.x[val_batch.batch_size:]
        
        # Get source and target nodes from edge_index
        src, dst = val_batch.edge_index
        
        # Only consider edges where dst is a validation node (< batch_size) and src is not (>= batch_size)
        valid_edges = (dst < val_batch.batch_size) & (src >= val_batch.batch_size)
        valid_src = src[valid_edges]
        valid_dst = dst[valid_edges]
        
        if len(valid_dst) > 0:  # Only proceed if we have valid edges
            # Count neighbors for validation nodes
            neighbor_counts = torch.zeros(val_batch.batch_size, device=val_batch.x.device)
            neighbor_counts.scatter_add_(0, valid_dst, torch.ones_like(valid_dst, dtype=torch.float))
            
            # Sum up neighbor features for validation nodes
            neighbor_sums = torch.zeros(val_batch.batch_size, val_batch.x.shape[1], device=val_batch.x.device)
            # Get features from source nodes
            src_features = val_batch.x[valid_src]
            # For each feature dimension, scatter add the source features to their respective destinations
            for i in range(val_batch.x.shape[1]):
                neighbor_sums[:, i].scatter_add_(0, valid_dst, src_features[:, i])
            
            # Compute means (avoiding division by zero)
            neighbor_counts = neighbor_counts.unsqueeze(-1).expand(-1, val_batch.x.shape[1])
            neighbor_counts = torch.where(neighbor_counts == 0, torch.ones_like(neighbor_counts), neighbor_counts)
            neighbor_means = neighbor_sums / neighbor_counts
            
            # Update only validation nodes with neighbor means
            mask_x[:val_batch.batch_size] = neighbor_means

        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch \
            = self(
                batch_x=val_batch.x,
                batch_edge_index=val_batch.edge_index,
                batch_encoder_conditions=val_encoder_conditions,
                batch_attr_decoder_conditions=val_attr_decoder_conditions,
                batch_adj_decoder_conditions=val_adj_decoder_conditions,
            )

        # prepare dictionary of data required for computing loss
        batch_size = val_batch.batch_size
        val_loss_data = {
                        'quantizer_input': h_latent[:batch_size],
                        'quantizer_output': h_quantized[:batch_size],
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
        Definition of a single test step of the VQNiche model on the current batch of nodes received from the test dataloader at the current training epoch.

        Parameters
        ----------
        - test_batch: torch_geometric.data.Data
            The input test data (batch of nodes).

        Returns
        -------
        - test_loss: torch.Tensor
            The computed loss for this batch.
        """
        # execute the forward of the VQNiche model
        test_encoder_conditions = getattr(test_batch, 'encoder_conditions', None)
        test_attr_decoder_conditions = getattr(test_batch, 'attr_decoder_conditions', None)
        test_adj_decoder_conditions = getattr(test_batch, 'adj_decoder_conditions', None)
        _, \
        _, \
        _, \
        _, \
        _, \
        _ \
            = self(
                batch_x=test_batch.x,
                batch_edge_index=test_batch.edge_index,
                batch_encoder_conditions=test_encoder_conditions,
                batch_attr_decoder_conditions=test_attr_decoder_conditions,
                batch_adj_decoder_conditions=test_adj_decoder_conditions,
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
            Dictionary containing inference data with keys: X, edge_index, H_latent, X_hat, H_adj, H_quantized, Indices
        """
        X = []
        Y_cell_types = []
        Y_niche_types = []
        XY_coordinates = []
        H_latent = []
        H_quantized = []
        Indices = []
        X_hat = []
        H_adj = []
        Logits = []
        
        for batch in dataloader:
            batch_size = batch.batch_size

            X.append(batch.x[:batch_size])
            Y_cell_types.append(batch.y[:batch_size])
            Y_niche_types.append(batch.y_niche_types[:batch_size])
            XY_coordinates.append(batch.xy_coordinates[:batch_size])

            if hasattr(batch, 'encoder_conditions'):
                batch_encoder_conditions = batch.encoder_conditions.to(self.device)
            else:
                batch_encoder_conditions = None

            if hasattr(batch, 'attr_decoder_conditions'):
                batch_attr_decoder_conditions = batch.attr_decoder_conditions.to(self.device)
            else:
                batch_attr_decoder_conditions = None

            if hasattr(batch, 'adj_decoder_conditions'):
                batch_adj_decoder_conditions = batch.adj_decoder_conditions.to(self.device)
            else:
                batch_adj_decoder_conditions = None

            if self.mask_strategy in ['zeros', 'learnable_parameter']:
                # mask all source nodes
                mask_ratio: float = 1.0
            elif self.mask_strategy == 'original':
                # do not mask any nodes
                mask_ratio = 0.0
            
            mask_idx: torch.BoolTensor = set_mask_indices(
                N=batch.x.size(0),
                batch_size=batch_size,
                mask_ratio=mask_ratio,
                deterministic=False,
            ).to(self.device)

            masked_x = self.prepare_masked_input(
                batch_x=batch.x,
                mask_idx=mask_idx,
            ).to(self.device)
            
            h_latent, \
            h_quantized, \
            indices, \
            xhat, \
            h_adj, \
            logits = self(
                batch_x=masked_x,
                batch_edge_index=batch.edge_index.to(self.device),
                batch_encoder_conditions=batch_encoder_conditions,
                batch_attr_decoder_conditions=batch_attr_decoder_conditions,
                batch_adj_decoder_conditions=batch_adj_decoder_conditions,
            )

            H_latent.append(h_latent[:batch_size])
            H_quantized.append(h_quantized[:batch_size])
            Indices.append(indices[:batch_size])
            X_hat.append(xhat[:batch_size])
            H_adj.append(h_adj[:batch_size])
            Logits.append(logits[:batch_size])

        X = torch.cat(X, dim=0)
        Y_cell_types = torch.cat(Y_cell_types, dim=0)
        Y_niche_types = torch.cat(Y_niche_types, dim=0)
        XY_coordinates = torch.cat(XY_coordinates, dim=0)
        H_latent = torch.cat(H_latent, dim=0)
        H_quantized = torch.cat(H_quantized, dim=0)
        Indices = torch.cat(Indices, dim=0)
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
            'H_quantized': H_quantized,
            'Indices': Indices,
            'codebook_size': self.encoder.vq.codebook_size,
            'separate': self.encoder.vq.separate_codebook_per_head,
            'num_heads': self.encoder.vq.heads,
        }