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
import torch.nn.functional as F

from .base_model import BaseModel
from ..encoders.vqniche_encoder import VQNiche_Encoder
from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features



class VQNiche(BaseModel):
    def __init__(
            self,
            model_name: Literal['VQNiche'] = 'VQNiche',
            imputation_params: dict = {},
            encoder_name: Literal['VQNiche_Encoder'] = 'VQNiche_Encoder',
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            adjacency_decoder_name: Literal['MLP_AdjacencyDecoder'] = 'MLP_AdjacencyDecoder',
            predictor_name: Literal['Linear'] = 'Linear',
            train_metrics_list: List[str] = [],
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

        self.imputation_params = imputation_params

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
        train_encoder_conditions = getattr(train_batch, 'encoder_conditions', None)
        train_attr_decoder_conditions = getattr(train_batch, 'attr_decoder_conditions', None)
        train_adj_decoder_conditions = getattr(train_batch, 'adj_decoder_conditions', None)

        if self.imputation_params['mask_strategy'] in ['zeros', 'learnable_parameter', 'learnable_embedding']:
        # if self.imputation_params['mask_strategy'] in ['learnable_parameter']:
            # 1) Annealed masking ratio for learnable token
            ratio = self.linear_anneal_mask_ratio(
                epoch=self.current_epoch,
                final_ratio=self.imputation_params['final_mask_ratio'],
                warmup_epochs=self.imputation_params['warmup_epochs'],
            )
            # Get deterministic flag from imputation params, default to False for backward compatibility
            deterministic = self.imputation_params.get('deterministic_masking', False)
            mask_idx = self.sample_center_mask(
                N=train_batch.x.size(0),
                batch_size=train_batch.batch_size,
                ratio=ratio,
                device=train_batch.x.device,
                deterministic=deterministic
            )  # only among center nodes
        else:
            mask_idx = None

        # 2) Prepare masked input (LEARNABLE)
        mask_x = self.set_mask(
            batch_x=train_batch.x,
            batch_edge_index=train_batch.edge_index,
            batch_size=train_batch.batch_size,
            mask_strategy=self.imputation_params['mask_strategy'],
            mask_idx=mask_idx,
        )
        
        # stats = self.masked_input_diversity(
        #     x_in=mask_x,
        #     batch_size=train_batch.batch_size,
        #     mask_idx=mask_idx,
        # )
        # print(stats)

        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch \
            = self(
                batch_x=mask_x,
                batch_edge_index=train_batch.edge_index,
                batch_encoder_conditions=train_encoder_conditions,
                batch_attr_decoder_conditions=train_attr_decoder_conditions,
                batch_adj_decoder_conditions=train_adj_decoder_conditions,
            )

        pred_attr = xhat_batch[mask_idx]
        target_attr = train_batch.x[mask_idx]

        # prepare dictionary of data required for computing loss
        # This slicing is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the target nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of target nodes is equal to the batch size. The remaining entries of the forward output are the logits for the sampled neighbors of the target nodes.
        batch_size = train_batch.batch_size
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

            mask_x = self.set_mask(
                batch_x=batch.x.to(self.device),
                batch_edge_index=batch.edge_index.to(self.device),
                batch_size=batch_size,
                mask_strategy=self.imputation_params['mask_strategy'],
            )

            h_latent, \
            h_quantized, \
            indices, \
            xhat, \
            h_adj, \
            logits = self(
                batch_x=mask_x.to(self.device),
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
        

    def linear_anneal_mask_ratio(
            self,
            epoch: int,
            final_ratio: float = 0.2,
            warmup_epochs: int = 10
        ) -> float:
        """Linearly increase mask ratio from 0.0 to final_ratio over warmup_epochs."""
        if warmup_epochs <= 0:
            return final_ratio
        return float(0.3 + final_ratio * min(epoch / warmup_epochs, 1.0))


    def sample_center_mask(
            self,
            N: int,
            batch_size: int,
            ratio: float,
            device: torch.device,
            deterministic: bool = False
        ) -> torch.BoolTensor:
        """Mask a subset of the FIRST batch_size nodes (center nodes).
        
        Args:
            N: Total number of nodes
            batch_size: Number of center nodes to consider for masking
            ratio: Fraction of nodes to mask
            device: Device to place tensors on
            deterministic: If True, uses deterministic stride-based masking.
                         If False, uses random masking (default).
        """
        m = torch.zeros(N, dtype=torch.bool, device=device)
        if ratio <= 0 or batch_size == 0:
            return m
            
        k = max(1, int(round(ratio * batch_size)))
        
        if deterministic:
            # Calculate stride to evenly space the masked nodes
            stride = max(1, batch_size // k)
            # Take evenly spaced indices up to k nodes
            idx = torch.arange(0, min(k * stride, batch_size), stride, device=device)
        else:
            # Random masking
            idx = torch.randperm(batch_size, device=device)[:k]
            
        m[idx] = True
        return m
    
    
    @torch.no_grad()
    def masked_input_diversity(
            self,
            x_in: torch.Tensor,
            batch_size: int,
            mask_idx: torch.Tensor | None = None,
            round_decimals: int = 6
        ) -> dict:
        """
        Quick diagnostics for input diversity among masked nodes.
        Returns a dict of simple stats; low values (or high cosine) indicate collapse.
        """
        N = x_in.size(0)
        if mask_idx is None:
            mask = torch.zeros(N, dtype=torch.bool, device=x_in.device)
            mask[:batch_size] = True           # NeighborLoader: centers are first `batch_size`
        else:
            mask = mask_idx if mask_idx.dtype == torch.bool else torch.zeros(
                N, dtype=torch.bool, device=x_in.device).scatter_(0, mask_idx.to(x_in.device), True)

        X = x_in[mask]                         # [M, F] masked inputs
        M = X.size(0)
        if M <= 1:
            return {"masked_count": int(M), "mean_feat_std": float('nan'),
                    "unique_row_frac": 1.0, "mean_pair_cos": float('nan'),
                    "p95_pair_cos": float('nan')}

        # 1) Per-feature std across masked nodes (0 => identical per feature)
        per_feat_std = X.std(dim=0)
        mean_feat_std = per_feat_std.mean().item()

        # 2) Unique rows fraction (≈0 if all rows equal)
        Xr = torch.round(X * (10**round_decimals)) / (10**round_decimals)
        unique_rows = torch.unique(Xr, dim=0).size(0)
        unique_row_frac = float(unique_rows) / float(M)

        # 3) Pairwise cosine similarity among masked rows (≈1 if identical)
        Xn = F.normalize(X, dim=1)
        cos = Xn @ Xn.T
        off = cos[~torch.eye(M, dtype=torch.bool, device=X.device)]
        mean_pair_cos = off.mean().item()
        p95_pair_cos = off.quantile(0.95).item()

        return {
            "masked_count": int(M),
            "mean_feat_std": mean_feat_std,
            "unique_row_frac": unique_row_frac,
            "mean_pair_cos": mean_pair_cos,
            "p95_pair_cos": p95_pair_cos,
        }