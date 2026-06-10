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
from ..modules import DispersionHead
from vqniche.utils.loss_utils import (
    batch_pred_attr_and_target_attr,
    aggregate_1hop_neighbor_features,
)

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
            test_metrics_list: List[str] = [],
            in_channels: int = None,
            out_channels: int = None,
            label_name: str = None,
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
        - label_name: str
            The name of the label.

        - train_metrics_list: List[str]
            The list of metrics to compute during training.
        - test_metrics_list: List[str]
            The list of metrics to compute during testing.

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
            test_metrics_list=test_metrics_list,
            **optimizer_params,
            **loss_params,
        )

        # Initialize imputation parameters
        self._init_imputation_params(imputation_params)
            
        # Initialize VQNiche encoder module.
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
        self.adjacency_decoder = self._init_adjacency_decoder(
            in_channels=self.encoder.dim,
            adjacency_decoder_name=adjacency_decoder_name,
            **adjacency_decoder_params,
        )
        print(f"3. Adjacency Decoder: {adjacency_decoder_name} that decodes {self.adjacency_decoder.in_channels} latent features to {self.adjacency_decoder.out_channels} adjacency features.")

        # Initialize the predictor.
        self.predictor = self._init_predictor(
                            predictor_name=predictor_name,
                            in_channels=self.encoder.dim,
                            out_channels=out_channels,
                        )
        print(f"4. Predictor: {predictor_name} that transforms {self.encoder.dim} hidden features to {out_channels} dimensional logits.")

        # ---- Adjacency-loss input mode (optional) --------------------------
        # By default, the BCE adjacency loss is computed on `h_adj`, which is
        # the output of the adjacency_decoder MLP fed with the quantized
        # latent. When `bypass_adj_decoder=True`, the BCE loss is computed
        # directly on the quantized latent z_q, matching the NicheCompass /
        # standard graph-VAE formulation:
        #     edge_logit_ij = z_q_i^T z_q_j      (no extra non-linearity)
        # This removes one MLP and forces the codebook embeddings themselves
        # to encode pairwise spatial proximity, which is exactly the bias
        # we want when the goal is for codes to capture spatial niches.
        # Read here so it is set BEFORE the forward path is exercised.
        loss_kwargs_for_flags = (loss_params.get('loss_kwargs', {})
                                 if loss_params else {})
        self.bypass_adj_decoder = bool(
            loss_kwargs_for_flags.get('bypass_adj_decoder', False)
        )

        # ---- Code-conditional dispersion (optional) ------------------------
        # When enabled, replace the global per-gene `dispersion` parameter
        # (kept on BaseModel as a fallback) with a small MLP that maps the
        # quantized embedding `z_q` to per-cell, per-gene log-dispersion.
        # The flag is read from loss_params['loss_kwargs']; default False.
        loss_kwargs = loss_params.get('loss_kwargs', {}) if loss_params else {}
        ccd_cfg = loss_kwargs.get('code_conditional_dispersion', False)
        if isinstance(ccd_cfg, bool):
            ccd_enabled = ccd_cfg
            ccd_hidden  = []
        else:
            # dict form: {'enabled': bool, 'hidden_channels': [..]}
            ccd_enabled = bool(ccd_cfg.get('enabled', False))
            ccd_hidden  = list(ccd_cfg.get('hidden_channels', []) or [])
        self.code_conditional_dispersion = ccd_enabled
        if ccd_enabled:
            self.dispersion_head = DispersionHead(
                in_channels=self.encoder.dim,
                out_channels=in_channels,
                hidden_channels=ccd_hidden,
            )
            print(
                f"5. Dispersion Head: code-conditional NB dispersion ENABLED "
                f"(z_q[{self.encoder.dim}] -> log_theta[{in_channels}]; hidden={ccd_hidden})."
            )
        else:
            self.dispersion_head = None
            print(f"5. Dispersion: GLOBAL per-gene parameter (code-conditional dispersion off).")

        # Initialize cache dicts for capturing data and model outputs during training, validation, and test steps
        self._init_inference_data_caches()


    def _init_imputation_params(
            self,
            imputation_params: dict,
        ) -> None:
        """
        Initialize imputation parameters for the model.

        Parameters
        ----------
        imputation_params : dict
            Dictionary containing imputation parameters:
            - mask_strategy: str, strategy for masking input features
            - base_mask_ratio: float, initial ratio of nodes to mask
            - final_mask_ratio: float, final ratio of nodes to mask
            - warmup_epochs: int, number of epochs for mask ratio warmup
            - deterministic_masking: bool, whether to use deterministic masking
            - compute_mask_input_diversity: bool, whether to compute mask input diversity
            - mask_token_eps: float, epsilon for mask token jittering
        """
        print(f"Setting imputation parameters:")
        for key, value in imputation_params.items():
            print(f"{key}: {value}")
        
        # Set imputation parameters as instance variables
        self.mask_strategy = imputation_params['mask_strategy']
        self.base_mask_ratio = imputation_params['base_mask_ratio']
        self.final_mask_ratio = imputation_params['final_mask_ratio']
        self.warmup_epochs = imputation_params['warmup_epochs']
        self.deterministic_masking = imputation_params['deterministic_masking']
        self.compute_mask_input_diversity = imputation_params['compute_mask_input_diversity']
        self.mask_token_eps = imputation_params['mask_token_eps']

        # Initialize learnable mask token if using learnable parameter strategy
        if self.mask_strategy == 'learnable_parameter':
            self.mask_token = torch.nn.Parameter(torch.empty(self.in_channels))
            torch.nn.init.normal_(self.mask_token, mean=2.0, std=1.0)


    def _init_inference_data_caches(self) -> None:
        """
        Initialize cache dictionaries for storing inference data during training, validation, and test steps.
        This includes setting up the data structure and adding VQ encoder metadata.
        """
        # Define cache keys for input data and model outputs
        data_keys = ['X', 'X_nbr', 'XY_coordinates', 'adata_batch_ids']
        # data_keys = ['X', 'X_nbr', 'XY_coordinates', 'Y_cell_types', 'Y_niche_types', 'adata_batch_ids']
        model_output_keys = ['H_latent', 'H_quantized', 'H_adj', 'Indices', 'X_hat', 'X_hat_nbr', 'Logits']
        self.cache_keys = data_keys + model_output_keys

        # Initialize empty caches for each phase
        self.train_inference_data_cache = {key: [] for key in self.cache_keys}
        self.val_inference_data_cache = {key: [] for key in self.cache_keys}
        self.test_inference_data_cache = {key: [] for key in self.cache_keys}

        # Add VQ encoder metadata to each cache
        for cache in [self.train_inference_data_cache, self.val_inference_data_cache, self.test_inference_data_cache]:
            cache['codebook_size'] = self.encoder.vq.codebook_size
            cache['separate'] = self.encoder.vq.separate_codebook_per_head
            cache['num_heads'] = self.encoder.vq.heads
            # Multi-level VQ metadata. Default to single-level for back-compat
            # with VectorQuantize and other lucidrains base classes.
            cache['num_quantizers'] = int(getattr(self.encoder.vq, 'num_quantizers', 1))
            # Per-level sizes (list); only set when the VQ exposes them.
            cb_sizes = getattr(self.encoder.vq, 'codebook_sizes', None)
            if cb_sizes is not None:
                cache['codebook_sizes'] = list(cb_sizes)


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor,
            batch_encoder_conditions: Optional[torch.Tensor] = None,
            batch_spatial_prior_features: Optional[torch.Tensor] = None,
            batch_attr_decoder_conditions: Optional[torch.Tensor] = None,
            batch_adj_decoder_conditions: Optional[torch.Tensor] = None,
            read_depth: Optional[torch.Tensor] = None,
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
        - batch_spatial_prior_features: torch.Tensor
            The spatial prior features for the encoder of the batch of nodes.
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
        - h_spatial_prior: torch.Tensor
            The spatial prior features for the encoder of the batch of nodes.

        Notes
        -----
        - h_latent is either the output of the MLP module (if no GNN layers are applied) or the output of the GNN module (if no MLP layers are applied) or the output of MLP followed by GNN modules (if both are applied).
        - h_quantized is the output of the VQ module.
        """
        # execute the forward of the VQNiche_Encoder module
        h_latent, \
        h_quantized, \
        indices, \
        h_spatial_prior \
            = self.encoder(
                            batch_x,
                            batch_edge_index,
                            batch_encoder_conditions,
                            batch_spatial_prior_features,
                        )

        # IMPORTANT: read_depth must be computed from the *un*masked input.
        # `batch_x` here is the masked tensor (callers pass `masked_x`), so
        # `batch_x.sum(dim=-1)` for masked source nodes equals the sum of the
        # mask token (~mask_token_eps * num_genes), which is meaningless as a
        # library size and breaks the NB target which is in raw counts.
        # Callers should now pass the un-masked `read_depth`; we fall back to
        # `batch_x.sum(dim=-1)` only for backward compatibility.
        if read_depth is None:
            read_depth = batch_x.sum(dim=-1)
        xhat = self.attribute_decoder(
                    x=h_quantized,
                    read_depth=read_depth,
                    conditions=batch_attr_decoder_conditions,
                )

        # decode the VQ-encoded edge embeddings to recover the adjacency matrix.
        # When `bypass_adj_decoder` is set, we feed the quantized latent z_q
        # *directly* into the BCE adjacency loss so the edge logits are raw
        # inner products z_q_i^T z_q_j — matching NicheCompass / standard
        # graph-VAE formulation. This is a strong inductive bias toward codes
        # whose embeddings cluster spatially.
        if self.bypass_adj_decoder:
            h_adj = h_quantized
        else:
            h_adj = self.adjacency_decoder(h_quantized)

        unnormalized_logits_batch = self.predictor(h_latent)

        return h_latent, \
            h_quantized, \
            indices, \
            xhat, \
            h_adj, \
            unnormalized_logits_batch, \
            h_spatial_prior


    # ----- helpers shared by training/validation/test step ------------------

    def _resolve_dispersion(self, h_quantized_seed: torch.Tensor) -> torch.Tensor:
        """
        Return the NB dispersion tensor for this batch.

        - If `code_conditional_dispersion=True`: a small MLP head maps the
          quantized embedding `z_q` (shape (B, D)) to per-cell, per-gene
          dispersion of shape (B, n_genes).
        - Otherwise: the legacy per-gene global parameter is exp'd to a
          shape-(n_genes,) tensor that broadcasts against (B, n_genes).
        """
        if self.code_conditional_dispersion and self.dispersion_head is not None:
            return self.dispersion_head(h_quantized_seed)
        return torch.exp(self.dispersion)

    def _safe_one_hot_indices(self, indices: torch.Tensor):
        """
        One-hot encode VQ indices for the spatial-prior loss.

        Multi-level VQ (RVQ / ConditionalVQ) emits indices of shape
        `(..., num_quantizers)` whose per-level codebooks have different
        sizes — there is no single `num_classes` to encode against. The
        spatial-prior loss is gated by `h_spatial_prior is not None`
        downstream, which is only set when an explicit spatial prior MLP
        is configured (none of the current ablation variants enable it),
        so we safely pass `None` in the multi-level case.
        """
        if int(getattr(self.encoder.vq, 'num_quantizers', 1)) > 1:
            return None
        return F.one_hot(indices, num_classes=self.encoder.vq.codebook_size)


    def apply_mask_to_attributes(
            self,
            batch_x: torch.Tensor,
            mask_idx: torch.LongTensor,
        ) -> torch.Tensor:
        """
        Mask the input attributes based on the mask indices.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - mask_idx: torch.LongTensor
            The indices of the nodes to mask.

        Returns
        -------
        - masked_x: torch.Tensor
            The masked input features.
        """
        # clone the input features to avoid modifying the original features
        masked_x = batch_x.clone()
        
        # if the mask strategy is zeros or learnable_parameter, mask the input features of the source nodes where mask_idx is True with the mask token
        if self.mask_strategy == 'learnable_parameter':
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
            # index[i] = j => the i-th row of mask is copied to the j-th row (dim=0) of masked_x
            masked_x.index_copy_(
                dim=0,
                index=index,
                source=mask,
            )
            
        return masked_x


    def prepare_masked_input(
            self,
            batch_size: int,
            batch_x: torch.Tensor,
            step: Literal['train', 'val', 'test', 'predict'] = 'train',
        ) -> torch.Tensor:
        """
        Prepare the masked input for the specified step.
        """
        # Set ratio of source nodes to mask
        if self.mask_strategy == 'learnable_parameter':
            # set number of source nodes to mask based on the mask strategy and mask ratio
            if step == 'train':
                mask_ratio: float = set_mask_ratio(
                                epoch=self.current_epoch,
                                base_ratio=self.base_mask_ratio,
                                final_ratio=self.final_mask_ratio,
                                warmup_epochs=self.warmup_epochs,
                            )
            # for validation and test, mask all source nodes
            else:
                mask_ratio = 1.0
        elif self.mask_strategy == 'original':
            # do not mask any nodes
            mask_ratio = 0.0
        
        # Set mask indices for the source nodes of the batch
        mask_idx: torch.LongTensor = set_mask_indices(
            N=batch_x.size(0),
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            deterministic=self.deterministic_masking
        )

        # Set the mask over the input features based on the mask strategy and mask indices
        # sampled neighbors of the source nodes are not masked
        masked_x = self.apply_mask_to_attributes(
            batch_x=batch_x,
            mask_idx=mask_idx,
        )

        # Compute and print the diversity of the masked input to measure if the mask token has collapsed or not
        # TODO: remove this after model is working
        if self.compute_mask_input_diversity:
            print_masked_input_diversity_stats(
                x_in=masked_x,
                batch_size=batch_size,
                mask_idx=mask_idx,
            )
            
        return mask_idx, masked_x


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
        
        # --------------------- Prepare Inputs for Training Forward Pass ---------------------
        # 1) Mask the input attributes based on the mask strategy and mask indices
        mask_idx, masked_x = self.prepare_masked_input(
                                batch_size=batch_size,
                                batch_x=train_batch.x,
                                step='train',
                            )

        # 2) Prepare conditioning features for the encoder, attribute decoder, and adjacency decoder
        train_encoder_conditions = getattr(train_batch, 'encoder_conditions', None)
        train_spatial_prior_features = getattr(train_batch, 'spatial_prior_features', None)
        train_attr_decoder_conditions = getattr(train_batch, 'attr_decoder_conditions', None)
        train_adj_decoder_conditions = getattr(train_batch, 'adj_decoder_conditions', None)

        # --------------------- Execute Forward Pass ---------------------
        # 3) Execute the forward pass of the VQNiche model
        # NOTE: pass the *un*masked read depth (library size) to the decoder so
        # that the softmax-by-read-depth decoder produces predictions with the
        # correct scale for the NB target (which is in raw counts of the
        # original, un-masked input).
        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch, \
        h_spatial_prior \
            = self(
                batch_x=masked_x,
                batch_edge_index=train_batch.edge_index,
                batch_encoder_conditions=train_encoder_conditions,
                batch_spatial_prior_features=train_spatial_prior_features,
                batch_attr_decoder_conditions=train_attr_decoder_conditions,
                batch_adj_decoder_conditions=train_adj_decoder_conditions,
                read_depth=train_batch.x.sum(dim=-1),
            )

        # --------------------- Prepare Data for Loss Computation ---------------------
        # 4) Prepare the predicted and target attributes for the Attribute Reconstruction Loss (negative binomial)
        #
        # `recon_mode` controls which NB target(s) are computed:
        #   "cell" — per-cell counts (legacy default, equivalent to k_hop_nb_loss=0)
        #   "nbr"  — 1-hop neighborhood mean counts (legacy k_hop_nb_loss=1)
        #   "both" — both, for use with TWO loss-name dispatches in the same step:
        #            `nb_attribute_reconstruction_loss` (uses pred_attr/target_attr,
        #             which point to the cell pair) AND
        #            `nb_attribute_reconstruction_loss_nbr` (uses
        #             pred_attr_nbr/target_attr_nbr).
        #
        # For backward compat, if `recon_mode` is not specified we infer it from
        # the legacy `k_hop_nb_loss` kwarg.
        recon_mode = self.loss_kwargs.get('recon_mode')
        if recon_mode is None:
            recon_mode = 'nbr' if self.loss_kwargs.get('k_hop_nb_loss', 0) == 1 else 'cell'
        if recon_mode not in ('cell', 'nbr', 'both'):
            raise ValueError(
                f"loss_kwargs['recon_mode'] must be one of "
                f"{{'cell', 'nbr', 'both'}}, got {recon_mode!r}."
            )

        pred_attr_nbr = target_attr_nbr = None
        if recon_mode in ('cell', 'both'):
            pred_attr, target_attr = batch_pred_attr_and_target_attr(
                batch_x=train_batch.x,
                batch_xhat=xhat_batch,
                edge_index=train_batch.edge_index,
                batch_size=batch_size,
                mask_idx=mask_idx,
                k_hop_nb_loss=0,
                only_masked=self.loss_kwargs['only_masked'],
            )
        if recon_mode in ('nbr', 'both'):
            pred_attr_nbr, target_attr_nbr = batch_pred_attr_and_target_attr(
                batch_x=train_batch.x,
                batch_xhat=xhat_batch,
                edge_index=train_batch.edge_index,
                batch_size=batch_size,
                mask_idx=mask_idx,
                k_hop_nb_loss=1,
                only_masked=self.loss_kwargs['only_masked'],
            )
        if recon_mode == 'nbr':
            # Legacy keys (used by `nb_attribute_reconstruction_loss`) point at
            # the neighborhood pair so existing variants that just set
            # k_hop_nb_loss=1 continue to work without code changes.
            pred_attr, target_attr = pred_attr_nbr, target_attr_nbr

        indices_one_hot = self._safe_one_hot_indices(indices)

        # 5) Prepare dictionary of data required for computing loss
        # Slicing the first batch_size entries is necessary because when the NeighborLoader (which wraps the NeighborSampler) is used, the source nodes, i.e. the nodes for which we compute the loss in this batch in this training step, are placed at the start of the batch. The number of source nodes is equal to the batch size. The remaining entries of the forward output correspond to the sampled neighbors of the source nodes.
        train_loss_data = {
                        'quantizer_input': h_latent[:batch_size], # code and commit loss
                        'quantizer_output': h_quantized[:batch_size], # code and commit loss
                        'pred_attr': pred_attr, # attribute reconstruction loss (cell or nbr depending on recon_mode)
                        'target_attr': target_attr, # attribute reconstruction loss
                        'edge_index': train_batch.edge_index, # attribute reconstruction loss
                        'batch_size': batch_size, # attribute and adjacency reconstruction loss
                        'dispersion': self._resolve_dispersion(h_quantized[:batch_size]), # attribute reconstruction loss
                        'h_adj': h_adj_batch, # adjacency reconstruction loss
                        'batch_edge_index': train_batch.edge_index, # adjacency reconstruction loss
                        # Per-node section ids — consumed by the BCE adjacency
                        # losses when `adj_within_section_only=True` (default)
                        # to drop cross-section pairs from the negative set.
                        # Set unconditionally; the dispatcher only reads it
                        # when the flag is on, so legacy variants are
                        # unaffected if no flag is set.
                        'node_adata_batch_ids': getattr(train_batch, 'adata_batch_ids', None),
                        'logits': unnormalized_logits_batch[:batch_size], # label prediction loss
                        'labels': train_batch.y[:batch_size], # label prediction loss
                        'h_spatial_prior': h_spatial_prior, # spatial prior loss
                        'indices_one_hot': indices_one_hot, # spatial prior loss
                        }
        if pred_attr_nbr is not None and recon_mode == 'both':
            # Only expose the nbr keys when we want them dispatched as a SECOND
            # loss term. In recon_mode='nbr' the nbr pair is already mapped onto
            # the legacy pred_attr/target_attr keys above.
            train_loss_data['pred_attr_nbr'] = pred_attr_nbr
            train_loss_data['target_attr_nbr'] = target_attr_nbr
        
        # Add mask token to train loss data if using learnable parameter strategy
        if self.mask_strategy == 'learnable_parameter':
            train_loss_data.update({
                'mask_token': self.mask_token,
            })

        # --------------------- Compute Loss ---------------------
        # 6) Compute the loss
        train_loss = self.common_step(
            batch_loss_data=train_loss_data,
            batch_size=batch_size,
            mode='train',
        )

        # --------------------- Cache Inference Data ---------------------
        # 7) Cache inference data
        self._cache_inference_data(
            batch=train_batch,
            batch_size=batch_size,
            h_latent=h_latent.detach(),
            h_quantized=h_quantized.detach(),
            h_adj_batch=h_adj_batch.detach(),
            indices=indices.detach(),
            xhat_batch=xhat_batch.detach(),
            unnormalized_logits_batch=unnormalized_logits_batch.detach(),
            cache_dict=self.train_inference_data_cache,
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
        
        # --------------------- Prepare Inputs for Validation Forward Pass ---------------------
        # 1) Mask the input attributes based on the mask strategy and mask indices
        mask_idx, masked_x = self.prepare_masked_input(
                                batch_size=batch_size,
                                batch_x=val_batch.x,
                                step='val',
                            )

        # 2) Prepare conditioning features for the encoder, attribute decoder, and adjacency decoder
        val_encoder_conditions = getattr(val_batch, 'encoder_conditions', None)
        val_spatial_prior_features = getattr(val_batch, 'spatial_prior_features', None)
        val_attr_decoder_conditions = getattr(val_batch, 'attr_decoder_conditions', None)
        val_adj_decoder_conditions = getattr(val_batch, 'adj_decoder_conditions', None)
        
        # --------------------- Execute Forward Pass ---------------------
        # 3) Execute the forward pass of the VQNiche model
        # NOTE: pass the *un*masked read depth (library size) — see training_step.
        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch, \
        h_spatial_prior \
            = self(
                batch_x=masked_x,
                batch_edge_index=val_batch.edge_index,
                batch_encoder_conditions=val_encoder_conditions,
                batch_spatial_prior_features=val_spatial_prior_features,
                batch_attr_decoder_conditions=val_attr_decoder_conditions,
                batch_adj_decoder_conditions=val_adj_decoder_conditions,
                read_depth=val_batch.x.sum(dim=-1),
            )
        
        # --------------------- Prepare Data for Loss Computation ---------------------
        # 4) Prepare NB targets — mirrors the recon_mode logic in training_step.
        recon_mode = self.loss_kwargs.get('recon_mode')
        if recon_mode is None:
            recon_mode = 'nbr' if self.loss_kwargs.get('k_hop_nb_loss', 0) == 1 else 'cell'

        pred_attr_nbr = target_attr_nbr = None
        if recon_mode in ('cell', 'both'):
            pred_attr, target_attr = batch_pred_attr_and_target_attr(
                batch_x=val_batch.x,
                batch_xhat=xhat_batch,
                edge_index=val_batch.edge_index,
                batch_size=batch_size,
                mask_idx=mask_idx,
                k_hop_nb_loss=0,
                only_masked=False,
            )
        if recon_mode in ('nbr', 'both'):
            pred_attr_nbr, target_attr_nbr = batch_pred_attr_and_target_attr(
                batch_x=val_batch.x,
                batch_xhat=xhat_batch,
                edge_index=val_batch.edge_index,
                batch_size=batch_size,
                mask_idx=mask_idx,
                k_hop_nb_loss=1,
                only_masked=False,
            )
        if recon_mode == 'nbr':
            pred_attr, target_attr = pred_attr_nbr, target_attr_nbr

        indices_one_hot = self._safe_one_hot_indices(indices)

        # 5) Prepare dictionary of data required for computing loss
        val_loss_data = {
                        'quantizer_input': h_latent[:batch_size],
                        'quantizer_output': h_quantized[:batch_size],
                        'pred_attr': pred_attr,
                        'target_attr': target_attr,
                        'edge_index': val_batch.edge_index,
                        'batch_size': batch_size,
                        'dispersion': self._resolve_dispersion(h_quantized[:batch_size]),
                        'h_adj': h_adj_batch,
                        'batch_edge_index': val_batch.edge_index,
                        'logits': unnormalized_logits_batch[:batch_size],
                        'labels': val_batch.y[:batch_size],
                        'h_spatial_prior': h_spatial_prior, # spatial prior loss
                        'indices_one_hot': indices_one_hot, # spatial prior loss
                        }
        if pred_attr_nbr is not None and recon_mode == 'both':
            val_loss_data['pred_attr_nbr'] = pred_attr_nbr
            val_loss_data['target_attr_nbr'] = target_attr_nbr

        # Add mask token to val loss data if using learnable parameter strategy
        if self.mask_strategy == 'learnable_parameter':
            val_loss_data.update({
                'mask_token': self.mask_token,
            })

        # --------------------- Compute Loss ---------------------
        # 6) Compute the loss
        val_loss = self.common_step(
            batch_loss_data=val_loss_data,
            batch_size=batch_size,
            mode='val',
        )
        
        # --------------------- Cache Inference Data ---------------------
        # 7) Cache inference data
        self._cache_inference_data(
            batch=val_batch,
            batch_size=batch_size,
            h_latent=h_latent,
            h_quantized=h_quantized,
            h_adj_batch=h_adj_batch,
            indices=indices,
            xhat_batch=xhat_batch,
            unnormalized_logits_batch=unnormalized_logits_batch,
            cache_dict=self.val_inference_data_cache,
        )
        
        return val_loss


    def test_step(
            self,
            test_batch: torch_geometric.data.Data
        ) -> None:
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
        batch_size = test_batch.batch_size
        
        # --------------------- Prepare Inputs for Test Forward Pass ---------------------
        # 1) Mask the input attributes based on the mask strategy and mask indices
        _, masked_x = self.prepare_masked_input(
                                batch_size=batch_size,
                                batch_x=test_batch.x,
                                step='test',
                            )

        # 2) Prepare conditioning features for the encoder, attribute decoder, and adjacency decoder
        test_encoder_conditions = getattr(test_batch, 'encoder_conditions', None)
        test_attr_decoder_conditions = getattr(test_batch, 'attr_decoder_conditions', None)
        test_adj_decoder_conditions = getattr(test_batch, 'adj_decoder_conditions', None)
        
        # --------------------- Execute Forward Pass ---------------------
        # 3) Execute the forward pass of the VQNiche model
        # NOTE: pass the *un*masked read depth (library size) — see training_step.
        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch, \
        _ \
            = self(
                batch_x=masked_x,
                batch_edge_index=test_batch.edge_index,
                batch_encoder_conditions=test_encoder_conditions,
                batch_attr_decoder_conditions=test_attr_decoder_conditions,
                batch_adj_decoder_conditions=test_adj_decoder_conditions,
                read_depth=test_batch.x.sum(dim=-1),
            )

        # Loss is not computed for test step
        
        # --------------------- Cache Inference Data ---------------------
        # 4) Cache inference data
        self._cache_inference_data(
            batch=test_batch,
            batch_size=batch_size,
            h_latent=h_latent,
            h_quantized=h_quantized,
            h_adj_batch=h_adj_batch,
            indices=indices,
            xhat_batch=xhat_batch,
            unnormalized_logits_batch=unnormalized_logits_batch,
            cache_dict=self.test_inference_data_cache,
        )
        
        return None


    def _cache_inference_data(
            self,
            batch: torch_geometric.data.Data,
            batch_size: int,
            h_latent: torch.Tensor,
            h_quantized: torch.Tensor,
            h_adj_batch: torch.Tensor,
            indices: torch.Tensor,
            xhat_batch: torch.Tensor,
            unnormalized_logits_batch: torch.Tensor,
            cache_dict: Optional[dict] = None,
        ) -> None:
        """
        Helper function to cache inference data during training, validation, and test steps.

        Parameters
        ----------
        cache_dict : dict
            The dictionary to store the cached data (train/val/test_inference_data_cache)
        batch : torch_geometric.data.Data
            The input batch data
        batch_size : int
            The batch size
        h_latent : torch.Tensor
            Latent node embeddings from the encoder
        h_quantized : torch.Tensor
            Quantized node embeddings from the encoder
        h_adj_batch : torch.Tensor
            Adjacency embeddings from the decoder
        indices : torch.Tensor
            Indices from vector quantization
        xhat_batch : torch.Tensor
            Reconstructed node attributes
        unnormalized_logits_batch : torch.Tensor
            Unnormalized logits from the predictor
        """
        if cache_dict is None:
            cache_dict = {key: [] for key in self.cache_keys}

        # Skip the appends if the caller is one of the persistent
        # train/val/test caches AND nobody is going to consume it.
        # `BaseModel.compute_metrics` only drains those caches at
        # epoch boundaries; with `train_metrics_list = []` (the
        # SQUINT_WITH_PEARSON=0 default) the consumer short-circuits
        # AND the per-step appends pile up GPU tensors until OOM.
        # Symptom: linear upward slope in the wandb GPU memory panel,
        # ending in `torch.cuda.OutOfMemoryError` inside the adjacency
        # BCE loss after ~75 s on a 140 GiB H100 at batch_size=1024.
        # `predict()` passes `cache_dict=None` and gets a fresh local
        # dict back — we always populate that (skip only fires for
        # the persistent caches).
        train_metrics_empty = not getattr(self, 'train_metrics_list', None)
        test_metrics_empty  = not getattr(self, 'test_metrics_list',  None)
        if cache_dict is getattr(self, 'train_inference_data_cache', None) and train_metrics_empty:
            return cache_dict
        if cache_dict is getattr(self, 'val_inference_data_cache',   None) and train_metrics_empty:
            return cache_dict
        if cache_dict is getattr(self, 'test_inference_data_cache',  None) and test_metrics_empty:
            return cache_dict

        cache_dict['X'].append(batch.x[:batch_size])
        cache_dict['X_nbr'].append(
            aggregate_1hop_neighbor_features(
                X=batch.x,
                edge_index=batch.edge_index,
                return_mean=True,
                batch_size=batch_size,
            )
        )

        # cache all labels
        for key in batch.keys():
            if key.startswith('y_'):
                if key not in cache_dict:
                    cache_dict[key] = []
                    if key not in self.cache_keys:
                        self.cache_keys.append(key)
                cache_dict[key].append(getattr(batch, key)[:batch_size])

        cache_dict['XY_coordinates'].append(batch.xy_coordinates[:batch_size])
        cache_dict['adata_batch_ids'].append(batch.adata_batch_ids[:batch_size])

        # Optional per-cell row index INTO the source AnnData's `.obs` —
        # used together with `adata_batch_ids` to look up arbitrary obs
        # columns from `dataset_blob.obs_per_batch_id` when building the
        # inference output AnnData. Same dynamic-cache pattern as `y_*`.
        if getattr(batch, 'obs_row_index', None) is not None:
            if 'obs_row_index' not in cache_dict:
                cache_dict['obs_row_index'] = []
                if 'obs_row_index' not in self.cache_keys:
                    self.cache_keys.append('obs_row_index')
            cache_dict['obs_row_index'].append(batch.obs_row_index[:batch_size])

        # Cache model outputs
        cache_dict['H_latent'].append(h_latent[:batch_size])
        cache_dict['H_quantized'].append(h_quantized[:batch_size])
        cache_dict['H_adj'].append(h_adj_batch[:batch_size])
        cache_dict['Indices'].append(indices[:batch_size])
        cache_dict['X_hat'].append(xhat_batch[:batch_size])
        cache_dict['X_hat_nbr'].append(
            aggregate_1hop_neighbor_features(
                X=xhat_batch,
                edge_index=batch.edge_index,
                return_mean=True,
                batch_size=batch_size,
            )
        )
        cache_dict['Logits'].append(unnormalized_logits_batch[:batch_size])
        
        return cache_dict
        
        
    def on_predict_model_eval(self) -> None:
        """
        Pytorch Lightning hook that is executed before the predict steps are called.
        """
        return super().on_predict_model_eval()
    
    
    def predict_step(
            self,
            predict_batch: torch_geometric.data.Data,
        ) -> torch.Tensor:
        """
        Definition of a single predict step of the VQNiche model on the current batch of nodes received from the predict dataloader at the current training epoch.
        """
        batch_size = predict_batch.batch_size
        
        # --------------------- Prepare Inputs for Test Forward Pass ---------------------
        # 1) Mask the input attributes based on the mask strategy and mask indices
        _, masked_x = self.prepare_masked_input(
                                batch_size=batch_size,
                                batch_x=predict_batch.x,
                                step='predict',
                            )

        # 2) Prepare conditioning features for the encoder, attribute decoder, and adjacency decoder
        predict_encoder_conditions = getattr(predict_batch, 'encoder_conditions', None)
        predict_attr_decoder_conditions = getattr(predict_batch, 'attr_decoder_conditions', None)
        predict_adj_decoder_conditions = getattr(predict_batch, 'adj_decoder_conditions', None)
        
        # --------------------- Execute Forward Pass ---------------------
        # 3) Execute the forward pass of the VQNiche model
        # NOTE: pass the *un*masked read depth (library size) — see training_step.
        h_latent, \
        h_quantized, \
        indices, \
        xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch, \
        _ \
            = self(
                batch_x=masked_x,
                batch_edge_index=predict_batch.edge_index,
                batch_encoder_conditions=predict_encoder_conditions,
                batch_attr_decoder_conditions=predict_attr_decoder_conditions,
                batch_adj_decoder_conditions=predict_adj_decoder_conditions,
                read_depth=predict_batch.x.sum(dim=-1),
            )

        # --------------------- Cache Inference Data ---------------------
        # 4) Cache inference data
        return self._cache_inference_data(
            batch=predict_batch,
            batch_size=batch_size,
            h_latent=h_latent,
            h_quantized=h_quantized,
            h_adj_batch=h_adj_batch,
            indices=indices,
            xhat_batch=xhat_batch,
            unnormalized_logits_batch=unnormalized_logits_batch,
        )
        
    
    def on_predict_epoch_end(self) -> None:
        """
        Pytorch Lightning hook that is executed after all predict steps are completed.
        """
        super().on_predict_epoch_end()