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
from vqniche.modules import CrossStitch
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
            niche_mlp_params: Optional[dict] = None,
            shared_mlp_params: Optional[dict] = None,
            cross_stitch_params: Optional[dict] = None,
            detach_gnn_input: bool = False,
        ):
        """
        Parameters
        ----------
        in_channels: int
            Number of input gene features.
        mlp_params: dict, optional
            Args for the SHARED / cell-branch MLP trunk. None disables
            the MLP, in which case the cell-VQ sees raw `x` directly
            (uncommon). When `niche_mlp_params` is also provided, this
            MLP feeds the cell branch ONLY (the niche branch then has
            its own MLP — see below). When `niche_mlp_params is None`
            (legacy default), this MLP is shared between cell + niche
            paths (legacy "shared trunk" behaviour).
        niche_mlp_params: dict, optional
            When provided, builds a SEPARATE niche-side MLP trunk —
            architecturally independent from `mlp_params`. The niche
            path becomes `x -> MLP_niche -> z_mlp_niche -> [niche_neck]
            -> GNN -> ...`, and `z_mlp_niche`'s gradients (from the
            adjacency BCE, nbr-NB, niche-commit) flow ONLY through
            `MLP_niche.weight`, NOT through `MLP_cell.weight`. The
            cell-VQ input (z_mlp_cell, from `mlp_params`) is then
            completely shielded from spatial-loss gradients — useful
            when the cell codebook is being pulled away from cell-
            type-discriminative geometry by spatial pressure on the
            shared trunk.
            Default `None` -> niche shares the cell-side MLP (legacy).
            For full decoupling at the same encoder shape, pass a
            deep copy of `mlp_params`.
        gnn_name: 'SAGEConv' | 'GATv2Conv' | 'GINConv'
            GNN aggregator class for the niche branch.
        gnn_params: dict
            Args for the GNN (hidden_channels, num_layers, etc.).
        conditioning_params: dict
            Optional FiLM conditioning, applied to the GNN output (so it
            affects the niche branch only). Use `condition_list=...` to
            enable; missing key disables it. When `niche_mlp_params`
            is set AND the cell/niche MLP output dims differ, FiLM
            will fail with a dim mismatch — keep niche MLP output dim
            equal to cell MLP output dim if you also enable FiLM.
        vq_cell_params: dict
            VQ class + args for the *cell* branch. `vq_name` selects the
            class (default 'VectorQuantize'). Quantises `z_mlp`.
        vq_niche_params: dict
            VQ class + args for the *niche* branch. Quantises `z_gnn`.
        cross_stitch_params: dict, optional
            When provided, inserts a cross-stitch unit (Misra et al.
            2016) that learns a 2x2 mix of the cell- and niche-branch
            post-MLP latents before the cell-VQ / GNN split — the
            "soft sharing" coupling that spans shared-to-decoupled.
            Requires the DECOUPLED architecture (`niche_mlp_params`
            set) and is mutually exclusive with the Y-shape shared
            trunk (`shared_mlp_params`). Keys are forwarded to
            `CrossStitch` (`per_channel`, `init_diag`, `init_off`).
            Default `None` -> no cross-stitch.
        detach_gnn_input: bool, default False
            When True, stop-gradient on the niche path input: the
            tensor fed into the (optional) niche_neck + GNN is detached
            so spatial-loss gradients never reach the MLP trunk that
            also produces the cell-VQ input. On the COUPLED (shared-
            trunk) architecture this is the gradient-level coupling
            variant — one physical trunk, trained only by the cell
            objective, with the niche modules learning on top of the
            detached features. Works with any trunk mode.
        """
        super().__init__()

        # ---- cell-side MLP trunk --------------------------------------------
        # When `niche_mlp_params` is None (legacy), this MLP is the
        # SHARED trunk feeding both branches. When `niche_mlp_params`
        # is provided, this is the cell-branch-only MLP.
        if mlp_params is None:
            self.mlp_module = None
            cell_in_channels = in_channels
        else:
            self.mlp_module = MLP_Module(in_channels=in_channels, **mlp_params)
            cell_in_channels = self.mlp_module.out_channels

        # ---- Optional SEPARATE niche-branch MLP trunk -----------------------
        # When `niche_mlp_params` is provided, builds a second MLP
        # module with independent weights. The niche path then uses
        # `niche_mlp_module(batch_x)` instead of the shared
        # `mlp_module(batch_x)`. Spatial gradients flow through
        # `niche_mlp_module.weight` ONLY; the cell-side MLP weights
        # are fully shielded from the adjacency BCE and nbr-NB
        # gradients.
        #
        # Output dim defaults to match `cell_in_channels` so the
        # downstream GNN / FiLM / niche-neck modules don't need
        # config changes. If you pass a `niche_mlp_params` whose
        # `hidden_channels[-1]` differs from the shared MLP's, you
        # must also tune `gnn_params` / `niche_neck_params` /
        # FiLM accordingly — and FiLM will fail if you don't.
        if niche_mlp_params is not None:
            self.niche_mlp_module = MLP_Module(
                in_channels=in_channels, **niche_mlp_params,
            )
            niche_post_mlp_dim = self.niche_mlp_module.out_channels
        else:
            self.niche_mlp_module = None
            niche_post_mlp_dim = cell_in_channels

        # ---- Optional SHARED MLP trunk (3-MLP Y-shape architecture) ---------
        # When `shared_mlp_params` is provided, builds a THIRD MLP
        # that processes `batch_x` independently. Its output is
        # CONCATENATED with each path-specific MLP's output before the
        # downstream cell-VQ and GNN modules:
        #
        #   x ─┬─► MLP_shared  ─► z_shared ──┐
        #      │                              │ concat
        #      ├─► MLP_cell    ─► z_path_cell ┴─► z_mlp_cell ─► VQ_cell
        #      │
        #      └─► MLP_niche   ─► z_path_niche ─┐
        #                                        ▼ concat with z_shared
        #                                       z_mlp_niche ─► GNN ─► VQ_niche
        #
        # Gradient routing:
        #   - Shared MLP receives gradients from BOTH branches
        #     (cell-NB + cell-commit AND spatial-adj + nbr-NB +
        #     niche-commit). It's where feature reuse happens.
        #   - Cell-path MLP receives ONLY cell-side gradients.
        #   - Niche-path MLP receives ONLY niche-side gradients.
        #
        # This is the "balanced" alternative between the two existing
        # modes:
        #   - Legacy (only `mlp_params` set): one shared MLP for both
        #     paths. Cell-VQ input is fully gradient-coupled to spatial
        #     losses via the shared trunk.
        #   - Decoupled (`mlp_params` + `niche_mlp_params`, no
        #     `shared_mlp_params`): two independent MLPs. Cell-VQ
        #     input is fully shielded from spatial gradients, but
        #     no shared feature reuse.
        #   - Y-shape (THIS mode, all three set): shared trunk for
        #     reused features, plus per-path MLPs for specialisation.
        #     Cell-path MLP weights are shielded from spatial
        #     gradients; the shared MLP weights are not.
        #
        # Sizing default (set by the helper
        # `_patch_dual_shared_specific_encoders`): each of the three
        # MLPs outputs HALF the original `mlp_params` final-layer dim.
        # E.g. original mlp_params=[400, 400, 256] -> each MLP becomes
        # [400, 400, 128]. The concatenated VQ_cell / GNN input stays
        # at 256 dim — same as the legacy shared-trunk architecture —
        # so the codebook embedding dim and downstream decoder shapes
        # are unchanged.
        #
        # When `shared_mlp_params is not None`, the user is expected
        # to also set `niche_mlp_params` (the helper does so by
        # default); if `niche_mlp_params is None` the niche path
        # uses the SHARED output ALONE, with no path-specific MLP —
        # asymmetric and probably not what was intended, but allowed.
        if shared_mlp_params is not None:
            self.shared_mlp_module = MLP_Module(
                in_channels=in_channels, **shared_mlp_params,
            )
            shared_out_dim = self.shared_mlp_module.out_channels
        else:
            self.shared_mlp_module = None
            shared_out_dim = 0

        # Effective dims downstream are the CONCATENATION of the
        # shared and path-specific outputs (with shared_out_dim=0
        # when shared is disabled).
        vq_cell_in_channels  = cell_in_channels  + shared_out_dim
        gnn_pre_neck_in_channels = niche_post_mlp_dim + shared_out_dim

        # ---- Optional CROSS-STITCH coupling (Misra et al. 2016) -------------
        # A learned 2x2 mixing of the cell- and niche-branch post-MLP
        # latents BEFORE the cell-VQ / GNN split:
        #
        #   z_mlp_cell'  = a_cc * z_mlp_cell + a_cn * z_mlp_niche
        #   z_mlp_niche' = a_nc * z_mlp_cell + a_nn * z_mlp_niche
        #
        # This is the "soft sharing" middle ground between the legacy
        # shared trunk (full coupling) and the decoupled encoders (no
        # coupling): a single cross-stitch unit spans the whole
        # shared-to-split spectrum and learns the optimal mix from data
        # (Misra et al., CVPR 2016), so the split point is not a
        # hand-tuned hyperparameter.
        #
        # Requirements (validated here):
        #   - DECOUPLED architecture: a separate `niche_mlp_module`
        #     must exist (so the two streams have independent params
        #     to mix). Cross-stitch on a single shared trunk is a
        #     no-op (both streams would be identical).
        #   - Mutually exclusive with the Y-shape shared trunk
        #     (`shared_mlp_params`) — mixing already-concatenated
        #     shared features is ill-defined here.
        #   - Equal cell / niche post-MLP dims (the 2x2 mixes matched
        #     channels). The decoupled patch makes the niche MLP a
        #     copy of the cell MLP, so this holds by default.
        if cross_stitch_params is not None:
            if self.niche_mlp_module is None:
                raise ValueError(
                    "cross_stitch_params requires the DECOUPLED encoder "
                    "(a separate niche_mlp_params); cross-stitch on a "
                    "single shared trunk is a no-op."
                )
            if self.shared_mlp_module is not None:
                raise ValueError(
                    "cross_stitch_params is mutually exclusive with the "
                    "Y-shape shared trunk (shared_mlp_params)."
                )
            if cell_in_channels != niche_post_mlp_dim:
                raise ValueError(
                    "cross_stitch requires equal cell/niche post-MLP "
                    f"dims, got cell={cell_in_channels} vs "
                    f"niche={niche_post_mlp_dim}. Match the niche MLP's "
                    "final hidden dim to the cell MLP's."
                )
            self.cross_stitch = CrossStitch(
                dim=cell_in_channels, **cross_stitch_params,
            )
        else:
            self.cross_stitch = None

        # ---- Optional STOP-GRADIENT on the niche (GNN) input ----------------
        # When True, the tensor fed into the niche path (GNN, via the
        # optional niche_neck) is `.detach()`-ed, so spatial-loss
        # gradients (adjacency BCE, nbr-NB, niche-commit) do NOT flow
        # back into the MLP trunk that also produces the cell-VQ input.
        # On the COUPLED (shared-trunk) architecture this is the
        # "gradient-level coupling" variant: one physical trunk shared
        # by both branches, but trained ONLY by the cell objective —
        # the GNN, niche VQ and niche decoder still train on top of the
        # detached features. Cheap (one detach), near-zero param
        # overhead. Independent of the architecture knobs above (works
        # with coupled, decoupled, or Y-shape trunks).
        self.detach_gnn_input = bool(detach_gnn_input)

        # ---- Optional NICHE NECK MLP (between z_mlp_niche and the GNN) ------
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
            # Niche-neck takes the (concatenated, when shared_mlp is
            # active) niche-side post-MLP output:
            #   - shared + niche-specific: `shared_out_dim +
            #     niche_post_mlp_dim`
            #   - decoupled (no shared):    `niche_post_mlp_dim`
            #   - shared-trunk only:        `cell_in_channels`
            # `gnn_pre_neck_in_channels` captures all three cases via
            # `niche_post_mlp_dim + shared_out_dim`, with
            # `shared_out_dim=0` for the modes that don't have a
            # shared MLP.
            self.niche_neck = MLP_Module(
                in_channels=gnn_pre_neck_in_channels,
                **niche_neck_params,
            )
            gnn_in_channels = self.niche_neck.out_channels
        else:
            self.niche_neck = None
            gnn_in_channels = gnn_pre_neck_in_channels

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
        #
        # FiLM dim policy by encoder mode:
        #   - Shared trunk / decoupled: FiLM acts on each path's post-
        #     MLP output, which has dim `cell_in_channels` (cell-side
        #     and niche-side MLPs match by convention; if user sets a
        #     different `niche_mlp_params['hidden_channels'][-1]`,
        #     they must keep it equal to the cell-side or disable
        #     FiLM — there's a single FiLM module).
        #   - Y-shape (shared_mlp_module set): FiLM acts on the SHARED
        #     output BEFORE concat with the path-specific MLPs. The
        #     path-specific MLPs DON'T see FiLM here. Rationale: the
        #     shared trunk is where batch-correlated features would
        #     come in (shared by both branches); the path-specific
        #     parts can stay unconditioned since they're branch-
        #     specialised. If you want FiLM on the path-specific
        #     outputs too, add a per-path FiLM downstream — not
        #     supported in this module yet.
        if 'condition_list' in conditioning_params:
            film_in_channels = (
                shared_out_dim if self.shared_mlp_module is not None
                else cell_in_channels
            )
            self.conditioning_module = FiLM(
                in_channels=film_in_channels,
                **conditioning_params,
            )
        else:
            self.conditioning_module = None

        # ---- VQ_cell --------------------------------------------------------
        # Quantizes the per-cell MLP output (pre-aggregation).
        # In Y-shape mode (shared_mlp_params set), the VQ_cell input
        # is `concat(z_shared, z_path_cell)`, so the codebook embedding
        # dim equals `cell_in_channels + shared_out_dim`. In the
        # legacy/decoupled modes `shared_out_dim=0` so this reduces to
        # the prior behaviour.
        vq_cell_params['dim'] = vq_cell_in_channels
        self.vq_cell = self._init_vq(vq_cell_params)

        # ---- VQ_niche -------------------------------------------------------
        # Quantizes the post-aggregation GNN output.
        vq_niche_params['dim'] = niche_in_channels
        self.vq_niche = self._init_vq(vq_niche_params)

        # Expose dims so the model can size its decoders correctly.
        # `cell_dim` is the dim of `z_q_cell` (= VQ_cell embedding
        # dim), which the cell decoder consumes.
        self.cell_dim  = vq_cell_in_channels
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
        # ---- shared MLP (Y-shape mode) --------------------------------------
        # If a shared MLP trunk is configured, compute it ONCE on
        # `batch_x` and apply FiLM (if any) here. The result will be
        # concatenated with each path-specific MLP's output below.
        if self.shared_mlp_module is not None:
            z_shared = self.shared_mlp_module(batch_x)
            if self.conditioning_module is not None:
                z_shared = self.conditioning_module(
                    x=z_shared, conditions=batch_encoder_conditions,
                )
        else:
            z_shared = None

        # ---- cell-side MLP --------------------------------------------------
        z_mlp_cell_path = (
            self.mlp_module(batch_x) if self.mlp_module is not None else batch_x
        )

        # FiLM on the cell-path output ONLY when the shared MLP is
        # NOT in play (legacy / decoupled modes). In Y-shape mode
        # FiLM was applied to `z_shared` already.
        if self.conditioning_module is not None and self.shared_mlp_module is None:
            z_mlp_cell_path = self.conditioning_module(
                x=z_mlp_cell_path, conditions=batch_encoder_conditions,
            )

        # Cell-VQ input: concat of [shared, cell-path] in Y-shape
        # mode; cell-path only otherwise. `torch.cat` along the last
        # axis since both tensors are (N, dim).
        if z_shared is not None:
            z_mlp = torch.cat([z_shared, z_mlp_cell_path], dim=-1)
        else:
            z_mlp = z_mlp_cell_path

        # ---- niche-side MLP -------------------------------------------------
        # If a separate niche MLP is configured (`niche_mlp_module is
        # not None`), the niche path runs its OWN MLP on `batch_x`
        # with INDEPENDENT weights — spatial-loss gradients flow only
        # through `niche_mlp_module.weight`, never reaching the
        # cell-side MLP. This is the "decoupled encoders" architecture.
        #
        # Default (`niche_mlp_module is None`): both branches share
        # the cell-side `z_mlp_cell_path` (legacy shared-trunk
        # behaviour) — the cell-VQ input is gradient-coupled to the
        # spatial losses via the shared MLP weights.
        if self.niche_mlp_module is not None:
            z_mlp_niche_path = self.niche_mlp_module(batch_x)
            # FiLM on the niche path only in NON-Y-shape mode
            # (mirrors the cell-path policy above).
            if self.conditioning_module is not None and self.shared_mlp_module is None:
                z_mlp_niche_path = self.conditioning_module(
                    x=z_mlp_niche_path, conditions=batch_encoder_conditions,
                )
        else:
            # No path-specific niche MLP. In legacy mode this means
            # niche reuses the cell-side output. In Y-shape mode WITH
            # shared MLP but no niche-specific MLP, niche gets just
            # the shared output (unusual, but the helper sets the
            # niche-specific MLP by default so this is rare).
            z_mlp_niche_path = z_mlp_cell_path

        # ---- cross-stitch mixing (Misra et al. 2016) ------------------------
        # Learned 2x2 combination of the two post-MLP (post-FiLM) path
        # latents. Only active in the decoupled architecture (validated
        # in __init__: cross_stitch requires a separate niche MLP and
        # forbids the Y-shape shared trunk), so `z_shared is None` here
        # and the cell-VQ input is exactly `z_mlp_cell_path`. We mix
        # both streams and re-bind `z_mlp` (the cell-VQ input, assembled
        # above from the un-mixed cell path) and the niche path.
        if self.cross_stitch is not None:
            z_mlp_cell_path, z_mlp_niche_path = self.cross_stitch(
                z_mlp_cell_path, z_mlp_niche_path,
            )
            z_mlp = z_mlp_cell_path        # z_shared is None in cross-stitch mode

        # Niche-pre-neck input: concat of [shared, niche-path] in
        # Y-shape mode; niche-path only otherwise.
        if z_shared is not None:
            z_mlp_niche = torch.cat([z_shared, z_mlp_niche_path], dim=-1)
        else:
            z_mlp_niche = z_mlp_niche_path

        # ---- niche-side pipeline: [niche_neck] -> GNN -> VQ_niche -----------
        # Stop-gradient coupling: when `detach_gnn_input` is set, the
        # niche path consumes a detached copy of `z_mlp_niche`, so
        # spatial-loss gradients never reach the (possibly shared) MLP
        # trunk that produces the cell-VQ input. The niche_neck / GNN /
        # niche VQ still train normally on the detached features.
        niche_path_input = (
            z_mlp_niche.detach() if self.detach_gnn_input else z_mlp_niche
        )
        gnn_input = (
            self.niche_neck(niche_path_input) if self.niche_neck is not None
            else niche_path_input
        )
        z_gnn = self.gnn_module(gnn_input, batch_edge_index)

        # Two independent quantizations. NOTE: cell-VQ takes
        # `z_mlp` (which is `concat(z_shared, z_mlp_cell_path)` in
        # Y-shape mode, or just `z_mlp_cell_path` otherwise) — never
        # the niche-side output. The whole point of decoupled / Y-
        # shape encoders is to keep the cell-VQ input free of niche-
        # branch gradient flow.
        z_q_cell,  idx_cell,  _ = self.vq_cell(z_mlp)
        z_q_niche, idx_niche, _ = self.vq_niche(z_gnn)

        return z_mlp, z_gnn, z_q_cell, z_q_niche, idx_cell, idx_niche

    def encoder_coupling_penalty(self) -> torch.Tensor:
        """
        Soft parameter-sharing penalty (Duong et al. 2015; Yang &
        Hospedales 2017) between the DECOUPLED cell- and niche-branch
        MLP trunks: the sum of squared differences between matched
        weight tensors,

            sum_l || W_cell[l] - W_niche[l] ||_F^2 .

        Minimising it (with a small weight, added to the total loss by
        the model) keeps the two independent trunks *similar* without
        physically sharing them — the lightest-touch coupling of the
        decoupled architecture. Requires both MLPs to exist with
        identical shapes (the decoupled patch makes the niche MLP a
        deep-copy of the cell MLP, so this holds by construction).

        Returns a scalar tensor on the module's device with grad to
        both trunks. Raises if the encoder is not decoupled.
        """
        if self.mlp_module is None or self.niche_mlp_module is None:
            raise RuntimeError(
                "encoder_coupling_penalty requires the DECOUPLED encoder "
                "(both a cell MLP `mlp_params` and a niche MLP "
                "`niche_mlp_params`). Got "
                f"mlp_module={self.mlp_module is not None}, "
                f"niche_mlp_module={self.niche_mlp_module is not None}."
            )
        cell_params = list(self.mlp_module.parameters())
        niche_params = list(self.niche_mlp_module.parameters())
        if len(cell_params) != len(niche_params):
            raise RuntimeError(
                "encoder_coupling_penalty requires matched cell/niche MLP "
                f"shapes; got {len(cell_params)} vs {len(niche_params)} "
                "parameter tensors."
            )
        pen = None
        for p_cell, p_niche in zip(cell_params, niche_params):
            if p_cell.shape != p_niche.shape:
                raise RuntimeError(
                    "encoder_coupling_penalty: mismatched parameter shapes "
                    f"{tuple(p_cell.shape)} vs {tuple(p_niche.shape)}."
                )
            term = ((p_cell - p_niche) ** 2).sum()
            pen = term if pen is None else pen + term
        return pen
