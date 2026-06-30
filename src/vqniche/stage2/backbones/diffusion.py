"""Discrete-diffusion backbone (absorbing-state / DiGress / D3PM family).

A noise-level-conditioned transformer denoiser behind the Backbone API. It is
architecturally the plain transformer backbone plus one ingredient: the body is
conditioned on the diffusion *timestep*, which here is proxied by the fraction
of cells currently masked. More masked == noisier == earlier in the reverse
process. Under an absorbing-state discrete diffusion, the held-out ("absorbed")
cells already carry the learnable ``[MASK]`` token, so the masked cells ARE the
noised tokens and predicting them is the standard denoising step.

The timestep is summarised per batch element as ``t in [0, 1]`` (masked
non-pad cells / total non-pad cells), embedded sinusoidally via
``timestep_embedding``, passed through a 2-layer MLP and added to every token
before the transformer blocks run.

Note: the MULTI-STEP diffusion *sampling* loop (iteratively denoising across a
schedule of mask-fraction / noise levels) is a decode-time concern handled
outside this module — this backbone only provides the timestep-conditioned
denoiser network for a single reverse step. Training reuses the existing
random-mask-fraction masked cross-entropy, which is exactly the absorbing-state
discrete-diffusion training objective (the variational bound reduces to a
reweighted cross-entropy over the absorbed tokens), so no extra loss is needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import Stage2Config
from ..transformer import TransformerBlock
from .base import Backbone, timestep_embedding


class DiffusionBackbone(Backbone):
    """Timestep(mask-fraction)-conditioned distance-biased transformer denoiser.

    Identical body to :class:`TransformerBackbone` (a stack of distance-biased
    self-attention blocks), with an additive per-token timestep embedding
    derived from the current mask fraction when ``cfg.model.diffusion_timestep_emb``
    is set.
    """

    def __init__(self, cfg: Stage2Config):
        super().__init__()
        mc = cfg.model
        self.d_model = mc.d_model
        self.use_timestep_emb = mc.diffusion_timestep_emb

        if self.use_timestep_emb:
            # Map the sinusoidal timestep embedding to the model dimension.
            self.t_mlp = nn.Sequential(
                nn.Linear(mc.d_model, mc.d_model),
                nn.SiLU(),
                nn.Linear(mc.d_model, mc.d_model),
            )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=mc.d_model,
                    n_heads=mc.n_heads,
                    d_ff=mc.d_ff,
                    dropout=mc.dropout,
                    dist_bias=mc.attn_dist_bias,
                    gamma_init=mc.attn_gamma_init,
                )
                for _ in range(mc.n_layers)
            ]
        )

    def _mask_fraction(
        self,
        x: torch.Tensor,                                   # (B, P, D)
        key_padding_mask: Optional[torch.Tensor],          # (B, P) True == PAD
        mask: Optional[torch.Tensor],                      # (B, P) True == held-out
    ) -> torch.Tensor:
        """Per-batch-element timestep ``t (B,)`` = masked non-pad / non-pad."""
        B = x.shape[0]
        if mask is None:
            return x.new_zeros(B)
        if key_padding_mask is not None:
            valid = ~key_padding_mask                      # (B, P) True == real cell
        else:
            valid = torch.ones_like(mask, dtype=torch.bool)
        n_masked = (mask & valid).sum(-1).float()          # (B,)
        n_valid = valid.sum(-1).clamp_min(1).float()       # (B,)
        return n_masked / n_valid

    def forward(self, x, coords, dist, key_padding_mask=None, mask=None):
        if self.use_timestep_emb:
            t = self._mask_fraction(x, key_padding_mask, mask)         # (B,)
            temb = timestep_embedding(t, self.d_model)                 # (B, D)
            temb = self.t_mlp(temb)                                    # (B, D)
            x = x + temb[:, None, :]                                   # broadcast over P

        for blk in self.blocks:
            x = blk(x, dist, key_padding_mask)
        return x
