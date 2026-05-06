"""
VQNiche_Dual: dual-codebook spatial transcriptomics model.

Architecture
------------
       x  (raw counts)
        │
        ▼
     shared MLP                ←  trains from BOTH branches' gradients
        │
       z_mlp ──────────────────────┐
        │                          │
        ▼                          ▼
     VQ_cell                    GNN(1+ layers, takes continuous z_mlp)
     (cell codebook)               │
        │                         z_gnn
       z_q_cell                    │
        │                          ▼
        ▼                       VQ_niche  (niche codebook)
   cell decoder                    │
   (NB on per-cell counts)        z_q_niche
                                    │
                                    ▼
                              niche decoder
                              (NB on neighborhood-mean counts)
                                    │
                                    └─── adjacency BCE on z_q_niche
                                         (cosine similarity, NicheCompass-style;
                                          NO MLP adjacency_decoder)

Key design choices (from D1–D6 in the design discussion)
--------------------------------------------------------
- (D1) Cell decoder takes only z_q_cell.
- (D2) Niche decoder takes only z_q_niche.
- (D3) Adjacency loss is `bce_cosine_adjacency_reconstruction_loss` directly
       on z_q_niche — no MLP adjacency_decoder.
- (D4) Both VQ slots accept any class registered in `get_vq_class`. Default
       is plain VectorQuantize (k=30 per branch); can be swapped to
       ResidualVQ_Squint or ConditionalVQ via config.
- (D5) Loss weights default to 1:1:1:1:1
       (NB_cell, NB_nbr, commit_cell, commit_niche, adj).
- (D6) Inference outputs:
       adata.obsm['cell_emb']           = z_q_cell
       adata.obsm['neighborhood_emb']   = z_q_niche
       adata.obs['cell_code_index']     or
       adata.obsm['cell_code_indices']  (depending on dim)
       adata.obs['neighborhood_code_index'] / .obsm['neighborhood_code_indices']
       adata.layers['X_hat']     = cell-decoder per-cell prediction
       adata.layers['X_hat_nbr'] = niche-decoder output, aggregated to nbr-mean
       adata.layers['X_nbr']     = ground-truth neighborhood mean

The two NB targets share the per-gene global `dispersion` parameter from
BaseModel; we don't yet expose code-conditional dispersion in this model
(the existing helper would need a per-branch head; can be added later).

Masking / FiLM / spatial-prior support: NOT included in v1 to keep the
implementation focused. The encoder accepts conditioning_params (FiLM)
because it inherits the existing FiLM module, but we don't thread the
masked-input pipeline through here. Add later if needed.
"""

from typing import Literal, List, Optional

import torch
import torch_geometric

from .base_model import BaseModel
from ..encoders.vqniche_dual_encoder import VQNiche_Dual_Encoder
from ..modules.adversary import BatchAdversaryHead
from vqniche.utils.loss_utils import (
    batch_pred_attr_and_target_attr,
    aggregate_1hop_neighbor_features,
)


class VQNiche_Dual(BaseModel):
    def __init__(
            self,
            model_name: Literal['VQNiche_Dual'] = 'VQNiche_Dual',
            encoder_name: Literal['VQNiche_Dual_Encoder'] = 'VQNiche_Dual_Encoder',
            attribute_decoder_name: Literal['MLPSoftmax'] = 'MLPSoftmax',
            adjacency_decoder_name: Optional[str] = None,    # not used
            predictor_name: Literal['Linear'] = 'Linear',
            train_metrics_list: List[str] = [],
            test_metrics_list: List[str] = [],
            in_channels: int = None,
            out_channels: int = None,
            label_name: Optional[str] = None,
            imputation_params: Optional[dict] = None,        # ignored (no masking yet)
            encoder_params: Optional[dict] = None,
            attribute_decoder_cell_params: Optional[dict]  = None,
            attribute_decoder_niche_params: Optional[dict] = None,
            adjacency_decoder_params: Optional[dict] = None,  # ignored
            optimizer_params: Optional[dict] = None,
            loss_params: Optional[dict] = None,
            decoder_covariate_dim: int = 0,
            adversarial_batch_dim: int = 0,
            adversarial_alpha: float = 1.0,
            adversarial_hidden_channels: Optional[List[int]] = None,
            adversarial_warmup_epochs: int = 0,
        ):
        # BaseModel.__init__ stores names + builds the loss-fn dispatcher.
        # We pass placeholder values for adjacency_decoder_name (unused)
        # because the dual model has no MLP adjacency decoder.
        super().__init__(
            model_name=model_name,
            encoder_name=encoder_name,
            attribute_decoder_name=attribute_decoder_name,
            adjacency_decoder_name=str(adjacency_decoder_name) if adjacency_decoder_name else "None",
            predictor_name=predictor_name,
            in_channels=in_channels,
            out_channels=out_channels,
            train_metrics_list=train_metrics_list,
            test_metrics_list=test_metrics_list,
            **(optimizer_params or {}),
            **(loss_params or {}),
        )

        # Encoder (contains both VQ branches)
        self.encoder = VQNiche_Dual_Encoder(
            in_channels=in_channels,
            **(encoder_params or {}),
        )
        print(
            f"1. VQNiche_Dual_Encoder: in={in_channels} -> "
            f"cell_dim={self.encoder.cell_dim}, niche_dim={self.encoder.niche_dim}; "
            f"vq_cell={type(self.encoder.vq_cell).__name__} "
            f"(k={self.encoder.vq_cell.codebook_size}), "
            f"vq_niche={type(self.encoder.vq_niche).__name__} "
            f"(k={self.encoder.vq_niche.codebook_size})."
        )

        # ---- decoder covariate (NicheCompass-style batch correction) -------
        # When `decoder_covariate_dim > 0`, the per-cell batch one-hot is
        # CONCATENATED with z_q before each decoder call. The decoders are
        # therefore built with extended input dims so the linear layers can
        # consume the joint (z_q, batch_one_hot) tensor. This is the
        # NicheCompass mechanism: the decoder uses the batch covariate to
        # explain per-batch gene patterns, freeing z (and the codebook) to
        # represent batch-invariant biological identity.
        # `train()` sets `decoder_covariate_dim` from `data_batch.encoder_
        # condition_dim` after data loading, so it's number-of-batches
        # (n_unique_samples) at runtime.
        self.decoder_covariate_dim = int(decoder_covariate_dim)

        cell_decoder_in  = self.encoder.cell_dim  + self.decoder_covariate_dim
        niche_decoder_in = self.encoder.niche_dim + self.decoder_covariate_dim

        # Two attribute decoders — separate weights, separate input dims.
        # Reuse the parent's _init_attribute_decoder for both.
        self.attribute_decoder_cell = self._init_attribute_decoder(
            in_channels=cell_decoder_in,
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=(attribute_decoder_cell_params or {}),
        )
        self.attribute_decoder_niche = self._init_attribute_decoder(
            in_channels=niche_decoder_in,
            out_channels=in_channels,
            attribute_decoder_name=attribute_decoder_name,
            attribute_decoder_params=(attribute_decoder_niche_params or {}),
        )
        cov_msg = (f" (+{self.decoder_covariate_dim} batch covariate dims)"
                   if self.decoder_covariate_dim > 0 else "")
        print(
            f"2. Cell decoder ({attribute_decoder_name}): "
            f"{cell_decoder_in} -> {in_channels}{cov_msg}."
        )
        print(
            f"3. Niche decoder ({attribute_decoder_name}): "
            f"{niche_decoder_in} -> {in_channels}{cov_msg}."
        )

        # Predictor (cell-type logits) on the continuous z_mlp — kept for
        # backward compat with the loss dispatcher's cross-entropy term, even
        # though the dual variants below don't include it by default.
        self.predictor = self._init_predictor(
            predictor_name=predictor_name,
            in_channels=self.encoder.cell_dim,
            out_channels=out_channels,
        )
        print(
            f"4. Predictor ({predictor_name}): {self.encoder.cell_dim} -> {out_channels}."
        )

        # NO adjacency_decoder — adjacency loss reads `z_q_niche` directly.
        # NO masking infrastructure for v1.
        self.mask_strategy = 'original'

        # ---- Adversarial batch-invariance head (optional) ------------------
        # When enabled (`adversarial_batch_dim > 0`), a small MLP on top of
        # z_mlp tries to predict the per-cell batch label. Its forward path
        # uses a Gradient Reversal Layer (GRL), so backprop through this
        # head pulls the encoder's parameters AWAY from being able to
        # encode batch info — providing the batch-invariance pressure
        # NicheCompass gets from its KL prior. The classifier itself
        # learns normally (the GRL is upstream of it).
        # Set in `train()` after data load (= n_distinct batches).
        if adversarial_batch_dim and adversarial_batch_dim > 0:
            self.batch_adversary = BatchAdversaryHead(
                in_channels=self.encoder.cell_dim,    # operates on z_mlp
                n_batches=adversarial_batch_dim,
                hidden_channels=adversarial_hidden_channels or [128],
                dropout=0.0,
            )
            self.adversarial_alpha = float(adversarial_alpha)
            print(
                f"6. Batch-adversary head: z_mlp[{self.encoder.cell_dim}] -> "
                f"{adversarial_batch_dim} batches "
                f"(hidden={adversarial_hidden_channels or [128]}, "
                f"alpha={adversarial_alpha})."
            )
        else:
            self.batch_adversary = None
            self.adversarial_alpha = 0.0

        # ---- separate NB dispersion for the niche branch -------------------
        # BaseModel already created `self.dispersion` (per-gene, learnable);
        # we use that for the cell-branch NB. The niche-branch NB target is
        # the 1-hop neighbourhood-mean of raw counts, which is ~8-9x smoother
        # than per-cell counts. With ONE shared dispersion, the niche-side
        # gradient (cleaner because the target is smoother) dominates and
        # drags theta up over training -> the cell-branch NB likelihood
        # degrades even when the cell decoder predictions are static. A
        # second per-gene parameter for the niche branch removes the
        # gradient conflict.
        self.dispersion_niche = torch.nn.Parameter(torch.randn(in_channels))

        self._init_inference_data_caches()

    # ------------------------------------------------------------------
    # Inference cache
    # ------------------------------------------------------------------

    def _init_inference_data_caches(self) -> None:
        # Tensors that will be torch.cat'd at the end of each epoch. The
        # optional per-cell `obs_row_index` (added together with
        # `adata_batch_ids` to form a unique per-cell key for arbitrary-
        # obs-column lookup at inference time) is added dynamically inside
        # `_cache_inference_data` only if the dataloader carries it — same
        # convention as the optional `y_*` label tensors below.
        data_keys = ['X', 'X_nbr', 'XY_coordinates', 'adata_batch_ids']
        cell_keys  = ['H_latent_cell',  'H_quantized_cell',  'Indices_cell',  'X_hat']
        niche_keys = ['H_latent_niche', 'H_quantized_niche', 'Indices_niche', 'X_hat_nbr']
        self.cache_keys = data_keys + cell_keys + niche_keys

        for split in ['train_inference_data_cache',
                      'val_inference_data_cache',
                      'test_inference_data_cache']:
            cache = {key: [] for key in self.cache_keys}

            # Per-branch metadata.
            cache['codebook_size_cell']     = self.encoder.vq_cell.codebook_size
            cache['codebook_size_niche']    = self.encoder.vq_niche.codebook_size
            cache['num_quantizers_cell']    = int(getattr(self.encoder.vq_cell,  'num_quantizers', 1))
            cache['num_quantizers_niche']   = int(getattr(self.encoder.vq_niche, 'num_quantizers', 1))
            cb_sizes_cell  = getattr(self.encoder.vq_cell,  'codebook_sizes', None)
            cb_sizes_niche = getattr(self.encoder.vq_niche, 'codebook_sizes', None)
            if cb_sizes_cell is not None:
                cache['codebook_sizes_cell']  = list(cb_sizes_cell)
            if cb_sizes_niche is not None:
                cache['codebook_sizes_niche'] = list(cb_sizes_niche)

            # Back-compat keys used by the existing benchmarking code:
            # the niche branch is the primary spatial signal so we expose its
            # metadata under the legacy single-codebook key names.
            cache['codebook_size']  = self.encoder.vq_niche.codebook_size
            cache['separate']       = self.encoder.vq_niche.separate_codebook_per_head
            cache['num_heads']      = self.encoder.vq_niche.heads
            cache['num_quantizers'] = int(getattr(self.encoder.vq_niche, 'num_quantizers', 1))
            if cb_sizes_niche is not None:
                cache['codebook_sizes'] = list(cb_sizes_niche)

            setattr(self, split, cache)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor,
            batch_encoder_conditions: Optional[torch.Tensor] = None,
            batch_attr_decoder_conditions: Optional[torch.Tensor] = None,
            read_depth: Optional[torch.Tensor] = None,
            adata_batch_ids: Optional[torch.Tensor] = None,
        ):
        """
        Returns
        -------
        z_mlp, z_gnn, z_q_cell, z_q_niche, idx_cell, idx_niche,
        xhat_cell, xhat_niche, logits
        """
        z_mlp, z_gnn, z_q_cell, z_q_niche, idx_cell, idx_niche = self.encoder(
            batch_x=batch_x,
            batch_edge_index=batch_edge_index,
            batch_encoder_conditions=batch_encoder_conditions,
        )

        if read_depth is None:
            read_depth = batch_x.sum(dim=-1)

        # NicheCompass-style decoder covariate: when enabled, the per-cell
        # batch one-hot is CONCATENATED with the quantized embedding before
        # each decoder. The decoder uses this batch indicator to fit per-
        # batch gene patterns, which leaves z_q (and the codebook) free to
        # represent batch-invariant biological identity. The decoders were
        # built with extended in_channels to consume the concat tensor.
        # Built directly from `adata_batch_ids` so this mechanism works
        # independently of (and is not interfered with by) any encoder-side
        # FiLM conditioning.
        if self.decoder_covariate_dim > 0:
            if adata_batch_ids is None:
                raise ValueError(
                    "decoder_covariate_dim > 0 but adata_batch_ids was not "
                    "passed to forward(). Pass `batch.adata_batch_ids` from "
                    "the step methods."
                )
            cov = torch.nn.functional.one_hot(
                adata_batch_ids.long(),
                num_classes=self.decoder_covariate_dim,
            ).float()
            z_q_cell_in  = torch.cat([z_q_cell,  cov], dim=-1)
            z_q_niche_in = torch.cat([z_q_niche, cov], dim=-1)
        else:
            z_q_cell_in  = z_q_cell
            z_q_niche_in = z_q_niche

        # Cell decoder: z_q_cell -> per-cell prediction at the cell's read depth.
        # `batch_attr_decoder_conditions` is threaded through to the decoder so
        # FiLM-conditioned variants (apply_conditioning='in-MLP' / 'pre-MLP')
        # can modulate each layer with the per-cell batch one-hot. With
        # `apply_conditioning=None` the decoder ignores `conditions`.
        xhat_cell = self.attribute_decoder_cell(
            x=z_q_cell_in,
            read_depth=read_depth,
            conditions=batch_attr_decoder_conditions,
        )
        # Niche decoder: z_q_niche -> per-cell prediction. We compute the NB
        # loss on the *aggregated* (1-hop neighborhood mean) version of this
        # output against the aggregated true counts — see training_step.
        xhat_niche = self.attribute_decoder_niche(
            x=z_q_niche_in,
            read_depth=read_depth,
            conditions=batch_attr_decoder_conditions,
        )

        logits = self.predictor(z_mlp)

        return (z_mlp, z_gnn, z_q_cell, z_q_niche,
                idx_cell, idx_niche,
                xhat_cell, xhat_niche, logits)

    # ------------------------------------------------------------------
    # Training / validation / test / predict steps
    # ------------------------------------------------------------------

    def _step(self, batch: torch_geometric.data.Data, mode: str):
        """
        Shared step body for train/val/predict. Mode controls only logging
        + which cache to write to.
        """
        batch_size = batch.batch_size

        encoder_conditions      = getattr(batch, 'encoder_conditions',      None)
        attr_decoder_conditions = getattr(batch, 'attr_decoder_conditions', None)
        adata_batch_ids         = getattr(batch, 'adata_batch_ids',         None)

        (z_mlp, z_gnn, z_q_cell, z_q_niche,
         idx_cell, idx_niche,
         xhat_cell, xhat_niche, logits) = self(
            batch_x=batch.x,
            batch_edge_index=batch.edge_index,
            batch_encoder_conditions=encoder_conditions,
            batch_attr_decoder_conditions=attr_decoder_conditions,
            read_depth=batch.x.sum(dim=-1),
            adata_batch_ids=adata_batch_ids,
        )

        # Cell-branch NB targets: per-cell counts, no aggregation.
        pred_attr, target_attr = batch_pred_attr_and_target_attr(
            batch_x=batch.x,
            batch_xhat=xhat_cell,
            edge_index=batch.edge_index,
            batch_size=batch_size,
            mask_idx=None,
            k_hop_nb_loss=0,
            only_masked=False,
        )
        # Niche-branch NB targets: K-hop neighborhood mean of the niche
        # decoder's per-cell output, against the K-hop neighborhood mean of
        # the true counts.
        #
        # `nbr_aggregation_hops` (default 1) controls the smoothing radius:
        #   K=1  -> exactly the previous behaviour (1-hop nbr-mean target)
        #   K=2  -> 2-hop smoothed target (broader spatial averaging,
        #           pushes the niche codebook to capture larger-scale
        #           spatial structure). For K>1 to be exact, the
        #           datamodule sampler must sample at least K-hop
        #           neighbours — otherwise the K-th aggregation operates on
        #           a partial neighbourhood and the result is approximate.
        #
        # We implement K-hop aggregation by applying 1-hop aggregation
        # iteratively (K-1) times to xhat_niche and x, then doing the
        # final 1-hop pass via batch_pred_attr_and_target_attr (which also
        # slices to `batch_size`).
        nbr_hops = int(self.loss_kwargs.get('nbr_aggregation_hops', 1) or 1)
        if nbr_hops < 1:
            raise ValueError(f"nbr_aggregation_hops must be >= 1, got {nbr_hops}")

        if nbr_hops == 1:
            xhat_niche_pre, x_pre = xhat_niche, batch.x
        else:
            xhat_niche_pre, x_pre = xhat_niche, batch.x
            for _ in range(nbr_hops - 1):
                xhat_niche_pre = aggregate_1hop_neighbor_features(
                    X=xhat_niche_pre,
                    edge_index=batch.edge_index,
                    return_mean=True,
                )
                x_pre = aggregate_1hop_neighbor_features(
                    X=x_pre,
                    edge_index=batch.edge_index,
                    return_mean=True,
                )

        pred_attr_nbr, target_attr_nbr = batch_pred_attr_and_target_attr(
            batch_x=x_pre,
            batch_xhat=xhat_niche_pre,
            edge_index=batch.edge_index,
            batch_size=batch_size,
            mask_idx=None,
            k_hop_nb_loss=1,
            only_masked=False,
        )

        # Adversarial batch-invariance head: predict batch from z_mlp;
        # GRL inside the head ensures the encoder is pushed AWAY from being
        # batch-predictive when this loss is back-propagated.
        #
        # We classify the FULL z_mlp tensor (seeds + sampled neighbours), not
        # just the seed-node prefix. The niche branch's GNN aggregates over
        # neighbour z_mlp activations to produce z_gnn, so to make z_gnn
        # batch-invariant we need every input to that aggregation — i.e.
        # every row of z_mlp — to be pushed toward batch-invariance, not
        # only the seed rows. Restricting the adversary to z_mlp[:batch_size]
        # leaves the encoder free to encode batch info in non-seed rows
        # which then leak into z_gnn via the GNN.
        batch_logits_for_loss = None
        if self.batch_adversary is not None:
            if adata_batch_ids is None:
                raise RuntimeError(
                    "VQNiche_Dual.batch_adversary is enabled but "
                    "`batch.adata_batch_ids` was not present on the data "
                    "batch — the data loader didn't propagate it. Check "
                    "that initialize_databatch is being called for this "
                    "variant."
                )
            batch_logits_for_loss = self.batch_adversary(
                z_mlp, alpha=self.adversarial_alpha,
            )

        loss_data = {
            # Cell NB
            'pred_attr':       pred_attr,
            'target_attr':     target_attr,
            # Niche NB
            'pred_attr_nbr':   pred_attr_nbr,
            'target_attr_nbr': target_attr_nbr,
            # Shared
            'edge_index':      batch.edge_index,
            'batch_edge_index': batch.edge_index,
            'batch_size':      batch_size,
            # Cell-branch NB reads `dispersion`; niche-branch NB reads
            # `dispersion_niche`. The two are decoupled per-gene parameters
            # so the niche-side gradient cannot drag the cell-branch theta
            # up (which was making the cell NB loss drift up over training
            # even when the cell decoder predictions weren't moving).
            'dispersion':       torch.exp(self.dispersion),
            'dispersion_niche': torch.exp(self.dispersion_niche),
            # Two commit losses (disjoint)
            'quantizer_input_cell':  z_mlp[:batch_size],
            'quantizer_output_cell': z_q_cell[:batch_size],
            'quantizer_input_niche': z_gnn[:batch_size],
            'quantizer_output_niche': z_q_niche[:batch_size],
            # Adjacency loss reads ONE of these two embeddings (cosine sim),
            # selected by `loss_kwargs['adj_loss_input']`:
            #   - 'z_gnn'      -> continuous, NicheCompass-faithful, default
            #   - 'z_q_niche'  -> quantized, opt-in
            # Pass FULL tensors (including sampled neighbours) so the BCE
            # has enough nodes for sampling positives + negatives.
            'z_gnn':           z_gnn,
            'z_q_niche':       z_q_niche,
            # Optional cross-entropy on cell-type prediction
            'logits':          logits[:batch_size],
            'labels':          getattr(batch, 'y', torch.zeros(batch_size, dtype=torch.long, device=batch.x.device))[:batch_size],
        }

        # Adversarial batch loss data (only when the adversary is built).
        # Loss dispatcher reads `batch_logits` and `batch_labels` keys.
        # FULL-batch labels (seeds + sampled neighbours) — paired 1:1 with
        # the FULL `z_mlp` we just classified above.
        if batch_logits_for_loss is not None:
            loss_data['batch_logits'] = batch_logits_for_loss
            loss_data['batch_labels'] = adata_batch_ids.long()

        loss_value = self.common_step(
            batch_loss_data=loss_data,
            batch_size=batch_size,
            mode=mode,
        )

        # Cache inference data for this split.
        cache_dict = getattr(self, f"{mode}_inference_data_cache", None)
        if cache_dict is not None:
            self._cache_inference_data(
                batch=batch,
                batch_size=batch_size,
                z_mlp=z_mlp.detach(),
                z_gnn=z_gnn.detach(),
                z_q_cell=z_q_cell.detach(),
                z_q_niche=z_q_niche.detach(),
                idx_cell=idx_cell.detach(),
                idx_niche=idx_niche.detach(),
                xhat_cell=xhat_cell.detach(),
                xhat_niche=xhat_niche.detach(),
                cache_dict=cache_dict,
            )

        return loss_value

    def training_step(self, train_batch, batch_idx: Optional[int] = None) -> torch.Tensor:
        return self._step(train_batch, mode='train')

    def validation_step(self, val_batch, batch_idx: Optional[int] = None) -> torch.Tensor:
        return self._step(val_batch, mode='val')

    def test_step(self, test_batch) -> None:
        # No loss computation in test_step; just cache.
        batch_size = test_batch.batch_size
        encoder_conditions      = getattr(test_batch, 'encoder_conditions',      None)
        attr_decoder_conditions = getattr(test_batch, 'attr_decoder_conditions', None)
        adata_batch_ids         = getattr(test_batch, 'adata_batch_ids',         None)
        (z_mlp, z_gnn, z_q_cell, z_q_niche,
         idx_cell, idx_niche,
         xhat_cell, xhat_niche, _logits) = self(
            batch_x=test_batch.x,
            batch_edge_index=test_batch.edge_index,
            batch_encoder_conditions=encoder_conditions,
            batch_attr_decoder_conditions=attr_decoder_conditions,
            read_depth=test_batch.x.sum(dim=-1),
            adata_batch_ids=adata_batch_ids,
        )
        self._cache_inference_data(
            batch=test_batch, batch_size=batch_size,
            z_mlp=z_mlp, z_gnn=z_gnn,
            z_q_cell=z_q_cell, z_q_niche=z_q_niche,
            idx_cell=idx_cell, idx_niche=idx_niche,
            xhat_cell=xhat_cell, xhat_niche=xhat_niche,
            cache_dict=self.test_inference_data_cache,
        )
        return None

    def predict_step(self, predict_batch) -> dict:
        """
        Forward + cache for a predict batch. Returns the per-batch cache so
        the trainer can stitch a final inference dict together.
        """
        batch_size = predict_batch.batch_size
        encoder_conditions      = getattr(predict_batch, 'encoder_conditions',      None)
        attr_decoder_conditions = getattr(predict_batch, 'attr_decoder_conditions', None)
        adata_batch_ids         = getattr(predict_batch, 'adata_batch_ids',         None)
        (z_mlp, z_gnn, z_q_cell, z_q_niche,
         idx_cell, idx_niche,
         xhat_cell, xhat_niche, _logits) = self(
            batch_x=predict_batch.x,
            batch_edge_index=predict_batch.edge_index,
            batch_encoder_conditions=encoder_conditions,
            batch_attr_decoder_conditions=attr_decoder_conditions,
            read_depth=predict_batch.x.sum(dim=-1),
            adata_batch_ids=adata_batch_ids,
        )
        # Build a per-batch cache the predict-collator can fold together.
        return self._cache_inference_data(
            batch=predict_batch, batch_size=batch_size,
            z_mlp=z_mlp, z_gnn=z_gnn,
            z_q_cell=z_q_cell, z_q_niche=z_q_niche,
            idx_cell=idx_cell, idx_niche=idx_niche,
            xhat_cell=xhat_cell, xhat_niche=xhat_niche,
            cache_dict=None,
        )

    def on_predict_model_eval(self) -> None:
        return super().on_predict_model_eval()

    def on_predict_epoch_end(self) -> None:
        super().on_predict_epoch_end()

    # ------------------------------------------------------------------
    # Inference-data cache
    # ------------------------------------------------------------------

    def _cache_inference_data(
            self,
            batch,
            batch_size,
            z_mlp,
            z_gnn,
            z_q_cell,
            z_q_niche,
            idx_cell,
            idx_niche,
            xhat_cell,
            xhat_niche,
            cache_dict: Optional[dict] = None,
        ) -> dict:
        if cache_dict is None:
            cache_dict = {key: [] for key in self.cache_keys}

        # ---- inputs / context ------------------------------------------------
        cache_dict['X'].append(batch.x[:batch_size])
        cache_dict['X_nbr'].append(
            aggregate_1hop_neighbor_features(
                X=batch.x,
                edge_index=batch.edge_index,
                return_mean=True,
                batch_size=batch_size,
            )
        )
        # Optional one-hot label tensors (named y_*) — propagate as-is
        # (matches the convention used in VQNiche._cache_inference_data).
        for key in batch.keys():
            if key.startswith('y_'):
                if key not in cache_dict:
                    cache_dict[key] = []
                    if key not in self.cache_keys:
                        self.cache_keys.append(key)
                cache_dict[key].append(getattr(batch, key)[:batch_size])
        cache_dict['XY_coordinates'].append(batch.xy_coordinates[:batch_size])
        cache_dict['adata_batch_ids'].append(batch.adata_batch_ids[:batch_size])
        # Optional per-cell row index INTO the source AnnData's `.obs`.
        # Only present when the dataset blob was built with the new
        # `process_anndata_batch` (and only flows through if the dataloader
        # carries it). Same dynamic-cache pattern as `y_*` below.
        if getattr(batch, 'obs_row_index', None) is not None:
            if 'obs_row_index' not in cache_dict:
                cache_dict['obs_row_index'] = []
                if 'obs_row_index' not in self.cache_keys:
                    self.cache_keys.append('obs_row_index')
            cache_dict['obs_row_index'].append(batch.obs_row_index[:batch_size])

        # ---- cell branch -----------------------------------------------------
        cache_dict['H_latent_cell'].append(z_mlp[:batch_size])
        cache_dict['H_quantized_cell'].append(z_q_cell[:batch_size])
        cache_dict['Indices_cell'].append(idx_cell[:batch_size])
        cache_dict['X_hat'].append(xhat_cell[:batch_size])

        # ---- niche branch ----------------------------------------------------
        cache_dict['H_latent_niche'].append(z_gnn[:batch_size])
        cache_dict['H_quantized_niche'].append(z_q_niche[:batch_size])
        cache_dict['Indices_niche'].append(idx_niche[:batch_size])
        # X_hat_nbr is the per-cell aggregated neighborhood mean of the niche
        # decoder's output — directly comparable to X_nbr (cached above).
        cache_dict['X_hat_nbr'].append(
            aggregate_1hop_neighbor_features(
                X=xhat_niche,
                edge_index=batch.edge_index,
                return_mean=True,
                batch_size=batch_size,
            )
        )

        return cache_dict
