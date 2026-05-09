"""
Hierarchical / multi-level Vector Quantization modules.

Two variants, both built atop lucidrains' battle-tested
`vector_quantize_pytorch.VectorQuantize` so the codebook EMA, dead-code
resampling, and STE behaviour are inherited rather than re-implemented.

ResidualVQ_Squint
    Sequential residual quantization (RQ-VAE style). Each level quantizes
    the residual r_k = z - sum_{i<k} c_i.detach(). The detach is critical
    so that downstream levels receive a non-zero straight-through gradient
    back to the encoder; it matches lucidrains' own ResidualVQ implementation.
    Final quantized output: z_q = sum_k c_k.

ConditionalVQ
    Tree-structured / hierarchical VQ. A coarse level-1 codebook (K1 codes)
    routes each cell to one of K1 separate level-2 codebooks (each K2 codes).
    Cells assigned to the same level-1 niche refine themselves within a
    bucket-specific codebook, giving a clean hierarchical interpretation
    (level 1 = coarse niche, level 2 = subtype within niche).
    Final quantized output: z_q = c_l1 + c_l2.

Both modules expose the same forward contract as `VectorQuantize`
    forward(z: (B, D)) -> (z_q: (B, D), indices: (B, num_quantizers), commit_loss)
plus the attribute set
    dim, codebook_size (scalar — the primary level-1 size, used for one-hot
    fallback / spatial-prior compat), codebook_sizes (list of per-level
    sizes), num_quantizers, heads=1, separate_codebook_per_head=False
so the encoder and downstream metric code can branch on `num_quantizers`
without further plumbing.
"""

from typing import Sequence, Union, List, Optional

import torch
import torch.nn as nn

from vector_quantize_pytorch import VectorQuantize


def _normalize_codebook_sizes(
        codebook_sizes: Union[int, Sequence[int]],
        num_quantizers: int,
    ) -> List[int]:
    """Accept either a scalar (broadcast to num_quantizers) or a sequence."""
    if isinstance(codebook_sizes, int):
        return [int(codebook_sizes)] * int(num_quantizers)
    out = [int(k) for k in codebook_sizes]
    assert len(out) == int(num_quantizers), (
        f"codebook_sizes length {len(out)} != num_quantizers {num_quantizers}"
    )
    return out


# ---------------------------------------------------------------------------
# Residual VQ (RQ-VAE style)
# ---------------------------------------------------------------------------

class ResidualVQ_Squint(nn.Module):
    """
    Residual Vector Quantization built as a stack of independent
    `VectorQuantize` modules with per-level codebook sizing.

    Forward semantics:
      residual = z
      for layer k in 0..L-1:
          c_k        = layer_k(residual)             # quantize
          residual   = residual - c_k.detach()       # remove explained part
      z_q = sum_k c_k

    Per-level commit losses returned by lucidrains are stacked and summed.
    EMA codebook updates happen inside each layer independently.

    Notes
    -----
    - The `.detach()` in the residual update is essential. Without it, the
      straight-through estimator on layer 0 collapses level-2 (and beyond)
      gradients to zero w.r.t. z because (z - z_q1) becomes detached under
      STE. With the detach, each level produces an independent identity
      gradient back to z, summing to L copies of the encoder gradient.
      This matches lucidrains' own ResidualVQ implementation.
    - kmeans_init is only enabled on level 0; deeper levels see residuals
      that are not directly clusterable from the data distribution and are
      better initialised randomly.
    - Dead-code resampling is enabled only on level 0 (>0 threshold). Deeper
      levels can have very sparse usage at init and would spuriously revive.
    """

    def __init__(
            self,
            dim: int,
            num_quantizers: int = 2,
            codebook_size: Union[int, Sequence[int]] = (30, 200),
            use_cosine_sim: bool = True,
            ema_update: bool = True,
            decay: float = 0.8,
            eps: float = 1e-5,
            threshold_ema_dead_code: int = 2,
            kmeans_init: bool = True,
            kmeans_iters: int = 10,
            sync_kmeans: bool = True,
            commitment_weight: float = 0.0,   # external commit loss is used
            sample_codebook_temp: float = 0.0,
        ):
        super().__init__()
        codebook_sizes = _normalize_codebook_sizes(codebook_size, num_quantizers)
        self.dim = dim
        self.codebook_sizes = codebook_sizes
        # Primary scalar exposed for one-hot / spatial-prior compat. Downstream
        # code that needs the full per-level list reads `codebook_sizes`.
        self.codebook_size = codebook_sizes[0]
        self.num_quantizers = len(codebook_sizes)
        self.heads = 1
        self.separate_codebook_per_head = False

        self.layers = nn.ModuleList([
            VectorQuantize(
                dim=dim,
                codebook_size=k,
                use_cosine_sim=use_cosine_sim,
                ema_update=ema_update,
                decay=decay,
                eps=eps,
                kmeans_init=(kmeans_init if i == 0 else False),
                kmeans_iters=kmeans_iters,
                sync_kmeans=sync_kmeans,
                threshold_ema_dead_code=(threshold_ema_dead_code if i == 0 else 0),
                commitment_weight=commitment_weight,
                sample_codebook_temp=sample_codebook_temp,
            )
            for i, k in enumerate(codebook_sizes)
        ])

    def forward(self, z: torch.Tensor):
        """
        z: (B, D)
        Returns:
            z_q: (B, D)                       sum of per-level codes
            indices: (B, num_quantizers)      one assignment per level
            commit_loss: (num_quantizers,)    per-level lucidrains loss; the
                                              external `mse_commit_loss` will
                                              also be applied on (h_latent,
                                              h_quantized) — these two are
                                              additive and both 0 when the
                                              external one is solely used.
        """
        residual = z
        z_q_total = torch.zeros_like(z)
        idx_list = []
        loss_list = []
        for layer in self.layers:
            z_q_l, idx_l, loss_l = layer(residual)
            z_q_total = z_q_total + z_q_l
            # Detach so deeper levels see a non-zero STE gradient back to z.
            residual = residual - z_q_l.detach()
            idx_list.append(idx_l)
            loss_list.append(loss_l if isinstance(loss_l, torch.Tensor)
                              else torch.zeros((), device=z.device, dtype=z.dtype))
        indices = torch.stack(idx_list, dim=-1)              # (B, L)
        commit_losses = torch.stack(loss_list)               # (L,)
        return z_q_total, indices, commit_losses


# ---------------------------------------------------------------------------
# Conditional / Tree VQ
# ---------------------------------------------------------------------------

class ConditionalVQ(nn.Module):
    """
    Tree-structured VQ with 2 or 3 levels.

    Level 1: a single VectorQuantize of size K1 (coarse niche).
    Level 2: K1 separate VectorQuantize modules, each of size K2 (one
             refinement codebook per level-1 code). Cells routed to the
             same level-1 code share one level-2 codebook.
    Level 3 (optional): K1 * K2 separate VectorQuantize modules, each
             of size K3 (one per (level-1, level-2) leaf). Activated by
             passing `codebook_size_l3` (else the module is 2-level and
             behaves exactly as before).

    Forward (3-level case):
      z_q1, idx_l1 = vq_l1(z)
      r1 = z - z_q1.detach()
      for c in 0..K1-1:
          mask = (idx_l1 == c)
          if any: z_q2[mask], idx_l2[mask] = vq_l2[c](r1[mask])
      r2 = r1 - z_q2.detach()
      for c in 0..K1-1, b in 0..K2-1:
          mask = (idx_l1 == c) & (idx_l2 == b)
          if any: z_q3[mask], idx_l3[mask] = vq_l3[c*K2 + b](r2[mask])
      z_q = z_q1 + z_q2 + z_q3

    Total expressivity: K1 × K2 (2-level) or K1 × K2 × K3 (3-level)
    distinct quantized representations.

    Per-bucket data sparsity grows with depth (a typical (l1, l2) leaf
    holds far fewer cells than an l1 bucket). Same defensive defaults
    as level 2 are reused for level 3:
      - kmeans_init=False (cannot reliably cluster a sparse leaf)
      - threshold_ema_dead_code=0 (don't revive dead codes when the
        leaf itself is small).
    """

    def __init__(
            self,
            dim: int,
            codebook_size_l1: int = 30,
            codebook_size_l2: int = 10,
            codebook_size_l3: Optional[int] = None,
            use_cosine_sim: bool = True,
            ema_update: bool = True,
            decay: float = 0.8,
            eps: float = 1e-5,
            threshold_ema_dead_code: int = 2,
            kmeans_init: bool = True,
            kmeans_iters: int = 10,
            sync_kmeans: bool = True,
            commitment_weight: float = 0.0,
            sample_codebook_temp: float = 0.0,
        ):
        super().__init__()
        self.dim = dim
        self.codebook_size_l1 = int(codebook_size_l1)
        self.codebook_size_l2 = int(codebook_size_l2)
        self.codebook_size_l3 = (
            int(codebook_size_l3) if codebook_size_l3 is not None else None
        )
        if self.codebook_size_l3 is not None:
            self.codebook_sizes = [
                self.codebook_size_l1,
                self.codebook_size_l2,
                self.codebook_size_l3,
            ]
            self.num_quantizers = 3
        else:
            self.codebook_sizes = [self.codebook_size_l1, self.codebook_size_l2]
            self.num_quantizers = 2
        self.codebook_size = self.codebook_size_l1   # primary (level 1)
        self.heads = 1
        self.separate_codebook_per_head = False

        # Level 1
        self.vq_l1 = VectorQuantize(
            dim=dim,
            codebook_size=self.codebook_size_l1,
            use_cosine_sim=use_cosine_sim,
            ema_update=ema_update,
            decay=decay,
            eps=eps,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            threshold_ema_dead_code=threshold_ema_dead_code,
            commitment_weight=commitment_weight,
            sample_codebook_temp=sample_codebook_temp,
        )
        # Level 2: one separate codebook per level-1 code.
        # Per-bucket data may be very sparse, so:
        #   - kmeans_init=False (cannot reliably cluster a sparse bucket)
        #   - threshold_ema_dead_code=0 (don't spuriously revive dead codes
        #     when the bucket itself is small)
        self.vq_l2 = nn.ModuleList([
            VectorQuantize(
                dim=dim,
                codebook_size=self.codebook_size_l2,
                use_cosine_sim=use_cosine_sim,
                ema_update=ema_update,
                decay=decay,
                eps=eps,
                kmeans_init=False,
                threshold_ema_dead_code=0,
                commitment_weight=commitment_weight,
                sample_codebook_temp=sample_codebook_temp,
            )
            for _ in range(self.codebook_size_l1)
        ])
        # Level 3: K1*K2 leaf codebooks, indexed flat as
        #   idx = l1 * K2 + l2
        # Stored as a single ModuleList (faster + simpler iteration than
        # nesting). Built only when 3-level is requested.
        if self.codebook_size_l3 is not None:
            self.vq_l3 = nn.ModuleList([
                VectorQuantize(
                    dim=dim,
                    codebook_size=self.codebook_size_l3,
                    use_cosine_sim=use_cosine_sim,
                    ema_update=ema_update,
                    decay=decay,
                    eps=eps,
                    kmeans_init=False,
                    threshold_ema_dead_code=0,
                    commitment_weight=commitment_weight,
                    sample_codebook_temp=sample_codebook_temp,
                )
                for _ in range(self.codebook_size_l1 * self.codebook_size_l2)
            ])
        else:
            self.vq_l3 = None

    def forward(self, z: torch.Tensor):
        """
        z: (B, D)
        Returns:
            z_q: (B, D)              sum_l z_q_l
            indices: (B, num_quantizers)
                                     [l1, l2]            (2-level)
                                     [l1, l2, l3]        (3-level)
            commit_loss: scalar      sum of per-level losses
                                     (mean across active buckets per
                                     level so empty buckets don't tilt
                                     the magnitude).
        """
        B, D = z.shape
        # Level 1
        z_q1, idx_l1, loss_l1 = self.vq_l1(z)
        if not isinstance(loss_l1, torch.Tensor):
            loss_l1 = torch.zeros((), device=z.device, dtype=z.dtype)

        # Residual for level 2; detach so level-2 STE has a clean path to z.
        r1 = z - z_q1.detach()

        # Level 2: gather, quantize per bucket, scatter
        z_q2 = torch.zeros_like(z_q1)
        idx_l2 = torch.zeros(B, dtype=idx_l1.dtype, device=z.device)
        loss_l2_terms: List[torch.Tensor] = []
        for c in range(self.codebook_size_l1):
            mask = (idx_l1 == c)
            n_in_bucket = int(mask.sum().item())
            if n_in_bucket == 0:
                continue
            r1_c = r1[mask]                                 # (n_c, D)
            z_q2_c, idx_l2_c, loss_l2_c = self.vq_l2[c](r1_c)
            z_q2[mask] = z_q2_c
            idx_l2[mask] = idx_l2_c
            if isinstance(loss_l2_c, torch.Tensor):
                loss_l2_terms.append(loss_l2_c)

        if len(loss_l2_terms) > 0:
            loss_l2 = torch.stack(loss_l2_terms).mean()
        else:
            loss_l2 = torch.zeros((), device=z.device, dtype=z.dtype)

        # ---- 2-level shortcut --------------------------------------------
        if self.vq_l3 is None:
            z_q = z_q1 + z_q2
            indices = torch.stack([idx_l1, idx_l2], dim=-1)     # (B, 2)
            commit_loss = loss_l1 + loss_l2
            return z_q, indices, commit_loss

        # ---- 3-level pass ------------------------------------------------
        # Detach for level-3 STE on the running residual.
        r2 = r1 - z_q2.detach()

        z_q3 = torch.zeros_like(z_q1)
        idx_l3 = torch.zeros(B, dtype=idx_l1.dtype, device=z.device)
        loss_l3_terms: List[torch.Tensor] = []
        K2 = self.codebook_size_l2
        # Iterate over (l1, l2) leaves. There are K1*K2 of them; we skip
        # any leaf with no cells (typical — most leaves are empty in any
        # given batch). The combined-key trick (`leaf_id = l1*K2 + l2`)
        # lets us hold all leaf VQs in a single ModuleList and avoids a
        # second nested module.
        for c in range(self.codebook_size_l1):
            mask_l1 = (idx_l1 == c)
            if not mask_l1.any():
                continue
            for b in range(K2):
                mask = mask_l1 & (idx_l2 == b)
                n_in_leaf = int(mask.sum().item())
                if n_in_leaf == 0:
                    continue
                leaf_id = c * K2 + b
                r2_cb = r2[mask]                            # (n_leaf, D)
                z_q3_cb, idx_l3_cb, loss_l3_cb = self.vq_l3[leaf_id](r2_cb)
                z_q3[mask] = z_q3_cb
                idx_l3[mask] = idx_l3_cb
                if isinstance(loss_l3_cb, torch.Tensor):
                    loss_l3_terms.append(loss_l3_cb)

        if len(loss_l3_terms) > 0:
            loss_l3 = torch.stack(loss_l3_terms).mean()
        else:
            loss_l3 = torch.zeros((), device=z.device, dtype=z.dtype)

        z_q = z_q1 + z_q2 + z_q3
        indices = torch.stack([idx_l1, idx_l2, idx_l3], dim=-1)  # (B, 3)
        commit_loss = loss_l1 + loss_l2 + loss_l3
        return z_q, indices, commit_loss
