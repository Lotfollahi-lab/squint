"""
Distance-biased graph transformer blocks (torch).

Each training/inference example is one spatial patch of <= P cells. Within a
patch we run FULL self-attention (cost O(P^2), bounded because P is the patch
size, typically ~1k) augmented with an additive spatial bias

    bias[h, i, j] = - softplus(gamma_h) * || x_i - x_j ||

where x are the per-patch normalised coordinates and gamma_h is a learnable
per-head positive scalar. This is the GPS / SpaGT "structure-reinforced
attention" idea: the model keeps a global receptive field (every cell can
attend to every other cell in the patch) while being softly biased toward
spatial locality, with the bias strength learned per head (some heads stay
local, others go global). Padded cells are masked out of the keys.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBiasedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        dist_bias: bool = True,
        gamma_init: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.dist_bias = dist_bias
        if dist_bias:
            # raw_gamma -> softplus -> positive per-head bias strength.
            # init so softplus(raw) ~= gamma_init.
            raw = torch.log(torch.expm1(torch.tensor(float(gamma_init))))
            self.raw_gamma = nn.Parameter(raw.repeat(n_heads))

    def forward(
        self,
        x: torch.Tensor,                       # (B, P, D)
        dist: Optional[torch.Tensor] = None,   # (B, P, P) pairwise distances
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True == PAD
    ) -> torch.Tensor:
        B, P, _ = x.shape
        qkv = self.qkv(x).reshape(B, P, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, P, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale     # (B, H, P, P)

        if self.dist_bias and dist is not None:
            gamma = F.softplus(self.raw_gamma).view(1, self.n_heads, 1, 1)
            scores = scores - gamma * dist.unsqueeze(1)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = scores.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                          # (B, H, P, hd)
        out = out.transpose(1, 2).reshape(B, P, self.d_model)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with graph-biased attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        dist_bias: bool = True,
        gamma_init: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GraphBiasedSelfAttention(
            d_model, n_heads, dropout, dist_bias, gamma_init
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dist=None, key_padding_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), dist, key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """(B, P, 2) -> (B, P, P) Euclidean distances (numerically safe)."""
    return torch.cdist(coords, coords, p=2.0)
