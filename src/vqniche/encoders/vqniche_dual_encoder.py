"""
Dual-codebook encoder for VQNiche_Dual.

Architecture:

    x (raw counts)
        │
        ▼
    shared MLP
        │
       z_mlp ───────────────────────┐
        │                           │
        ▼                           ▼
     VQ_cell                     GNN(1-hop+, takes z_mlp continuous)
        │                           │
       z_q_cell                  z_gnn
        │                           │
        ▼                           ▼
    cell decoder                VQ_niche
                                   │
                                z_q_niche
                                   │
                                   ▼
                                niche decoder

The cell branch quantizes the per-cell features that the MLP extracted
*before* any neighbourhood aggregation — so VQ_cell has no architectural
access to neighbours' information and is structurally biased to encode
cell-intrinsic signal (e.g. cell type).

The niche branch quantizes the post-aggregation features from the GNN, so
VQ_niche only sees signal that has been mixed with neighbours' features —
structurally biased to encode niche / spatial-context signal.

Both VQ slots accept any class registered in `get_vq_class` (single
`VectorQuantize`, hierarchical `ResidualVQ_Squint`, tree `ConditionalVQ`,
etc.).  The two slots have *independent* `vq_cell_params` and
`vq_niche_params` dicts so they can be configured separately.

The GNN consumes the **continuous** `z_mlp`, not `z_q_cell`.  Feeding the
GNN the discretised cell code would be a severe bottleneck (only K_cell
distinct inputs per cell) and would couple the two branches in a way that
breaks the architectural disentanglement.
"""

from typing import Literal, Optional

import torch
import pytorch_lightning as pl

from vqniche.modules import MLP as MLP_Module
from vqniche.modules import init_gnn_module
from vqniche.modules import FiLM
from vqniche.modules import get_vq_class, get_valid_params


class VQNiche_Dual_Encoder(pl.LightningModule):
    def __init__(
            self,
            in_channels: int = None,
            mlp_params: Optional[dict] = None,
            gnn_name: Optional[Literal['SAGEConv', 'GATv2Conv', 'GINConv']] = None,
            gnn_params: dict = {},
            conditioning_params: dict = {},
            vq_cell_params: dict = {},
            vq_niche_params: dict = {},
            niche_neck_params: Optional[dict] = None,
        ):
        """
        Parameters
        ----------
        in_channels: int
            Number of input gene features.
        mlp_params: dict, optional
            Args for the shared MLP trunk. None disables the MLP, in which
            case both VQ branches see the raw `x` directly (uncommon).
        gnn_name: 'SAGEConv' | 'GATv2Conv' | 'GINConv'
            GNN aggregator class for the niche branch.
        gnn_params: dict
            Args for the GNN (hidden_channels, num_layers, etc.).
        conditioning_params: dict
            Optional FiLM conditioning, applied to the GNN output (so it
            affects the niche branch only). Use `condition_list=...` to
            enable; missing key disables it.
        vq_cell_params: dict
            VQ class + args for the *cell* branch. `vq_name` selects the
            class (default 'VectorQuantize'). Quantises `z_mlp`.
        vq_niche_params: dict
            VQ class + args for the *niche* branch. Quantises `z_gnn`.
        """
        super().__init__()

        # ---- shared MLP trunk ------------------------------------------------
        if mlp_params is None:
            self.mlp_module = None
            cell_in_channels = in_channels
        else:
            self.mlp_module = MLP_Module(in_channels=in_channels, **mlp_params)
            cell_in_channels = self.mlp_module.out_channels

        # ---- Optional NICHE NECK MLP (between z_mlp and the GNN) ------------
        # When `niche_neck_params` is provided, an additional MLP is
        # inserted on the niche-side path BETWEEN z_mlp and the GNN:
        #
        #   z_mlp  ──────► VQ_cell  (unchanged — cell-VQ still sees z_mlp)
        #     │
        #     ▼
        #   niche_neck
        #     │
        #     ▼
        #   GNN ──► VQ_niche
        #
        # This decouples the niche-side input from `z_mlp`: the GNN can
        # learn a different "view" of z_mlp without disturbing the cell
        # quantiser's commitment. Useful for testing whether the cell VQ
        # was being pulled away from cell-type-discriminative geometry
        # by gradients flowing back from the niche branch through the
        # shared `z_mlp`.
        #
        # Default `None` -> niche_neck is identity (legacy behaviour:
        # GNN consumes z_mlp directly). The output dim of `niche_neck`
        # becomes the GNN's input dim — usually we keep it equal to
        # `cell_in_channels` so the GNN module config doesn't need to
        # change.
        if niche_neck_params is not None:
            self.niche_neck = MLP_Module(
                in_channels=cell_in_channels,
                **niche_neck_params,
            )
            gnn_in_channels = self.niche_neck.out_channels
        else:
            self.niche_neck = None
            gnn_in_channels = cell_in_channels

        # ---- GNN (consumes continuous z_mlp OR niche_neck(z_mlp)) -----------
        if not gnn_params or gnn_params.get('num_layers', 0) == 0:
            raise ValueError(
                "VQNiche_Dual_Encoder requires at least one GNN layer; the "
                "niche branch is defined by post-GNN features. Got "
                f"gnn_params={gnn_params}."
            )
        self.gnn_module = init_gnn_module(
            in_channels=gnn_in_channels,
            gnn_name=gnn_name,
            gnn_params=gnn_params,
        )
        niche_in_channels = self.gnn_module.dim

        # ---- optional FiLM batch conditioning -------------------------------
        # Applied to the post-MLP latent (z_mlp), BEFORE the split into the
        # cell branch (VQ_cell directly) and the niche branch (GNN ->
        # VQ_niche). This way both codebooks see a batch-corrected
        # representation, so cells from different platforms/replicates with
        # the same biological identity end up assigned to the same code.
        # Conditioning vector is per-cell (built upstream by the data loader
        # from `adata.obs['batch']` or cell-id parsing), one-hot encoded.
        if 'condition_list' in conditioning_params:
            self.conditioning_module = FiLM(
                in_channels=cell_in_channels,        # FiLM at MLP output
                **conditioning_params,
            )
        else:
            self.conditioning_module = None

        # ---- VQ_cell --------------------------------------------------------
        # Quantizes the per-cell MLP output (pre-aggregation).
        vq_cell_params['dim'] = cell_in_channels
        self.vq_cell = self._init_vq(vq_cell_params)

        # ---- VQ_niche -------------------------------------------------------
        # Quantizes the post-aggregation GNN output.
        vq_niche_params['dim'] = niche_in_channels
        self.vq_niche = self._init_vq(vq_niche_params)

        # Expose dims so the model can size its decoders correctly.
        self.cell_dim  = cell_in_channels
        self.niche_dim = niche_in_channels

    @staticmethod
    def _init_vq(vq_params: dict):
        VQ_Module = get_vq_class(vq_params['vq_name'])
        valid = get_valid_params(VQ_Module, vq_params)
        return VQ_Module(**valid)

    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor,
            batch_encoder_conditions: Optional[torch.Tensor] = None,
        ):
        """
        Parameters
        ----------
        batch_x: (N, n_genes)
        batch_edge_index: (2, num_edges)  — local node-IDs of the batch
        batch_encoder_conditions: optional FiLM conditioning tensor.

        Returns
        -------
        z_mlp:      (N, cell_dim)        continuous, post-MLP, pre-aggregation
        z_gnn:      (N, niche_dim)       continuous, post-GNN
        z_q_cell:   (N, cell_dim)        VQ-quantized z_mlp
        z_q_niche:  (N, niche_dim)       VQ-quantized z_gnn
        idx_cell:   (N,) or (N, Q)       cell-codebook indices
        idx_niche:  (N,) or (N, Q)       niche-codebook indices
        """
        # MLP (shared trunk)
        z_mlp = self.mlp_module(batch_x) if self.mlp_module is not None else batch_x

        # FiLM applied at the MLP output (post-MLP, pre-VQ-cell, pre-GNN) so
        # both branches inherit a batch-corrected representation and the
        # cell + niche codebooks become batch-invariant.
        if self.conditioning_module is not None:
            z_mlp = self.conditioning_module(
                x=z_mlp, conditions=batch_encoder_conditions,
            )

        # GNN consumes the conditioned (continuous) z_mlp, optionally
        # routed through a niche-side neck MLP first (see __init__).
        # When `self.niche_neck is None` (default), this is identical
        # to the legacy behaviour `z_gnn = self.gnn_module(z_mlp, ...)`.
        gnn_input = (
            self.niche_neck(z_mlp) if self.niche_neck is not None else z_mlp
        )
        z_gnn = self.gnn_module(gnn_input, batch_edge_index)

        # Two independent quantizations. NOTE: cell-VQ takes `z_mlp`
        # directly (NOT the niche_neck output) — the whole point of
        # niche_neck is to decouple the niche path from the cell-VQ
        # input.
        z_q_cell,  idx_cell,  _ = self.vq_cell(z_mlp)
        z_q_niche, idx_niche, _ = self.vq_niche(z_gnn)

        return z_mlp, z_gnn, z_q_cell, z_q_niche, idx_cell, idx_niche
