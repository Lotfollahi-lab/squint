"""
Backbone interface + shared helpers for the stage-2 architecture ablation.

Every backbone is an ``nn.Module`` mapping embedded tokens -> per-cell hidden
states, with this EXACT signature:

    forward(x, coords, dist, key_padding_mask=None, mask=None) -> h

      x      (B, P, D)  embedded input tokens (code embeddings + positional
                        encoding; held-out cells already carry the model's
                        learnable [MASK] token, added by the caller)
      coords (B, P, 2)  per-patch normalised coordinates
      dist   (B, P, P)  precomputed pairwise Euclidean distances
      key_padding_mask (B, P) bool, True == PAD slot (not a real cell)
      mask   (B, P)     bool, True == held-out cell to predict
                        (graphmae / diffusion use it; others may ignore it)

    returns h (B, P, D)

The parent ``SpatialCodeTransformer`` applies a final LayerNorm and the
per-(branch, level) prediction heads on top of ``h`` — so a backbone must NOT
apply the heads or a final norm, and must NOT change P or the cell ordering.
Keeping this contract means the embedding, heads, loss and MaskGIT decode are
all backbone-agnostic and the archs are a clean head-to-head ablation.

`knn_adjacency` builds a row-normalised spatial-kNN adjacency from the dense
distance matrix (no torch_geometric dependency; patches are ~1k cells so dense
top-k is cheap). Use ``A @ x`` for mean message passing / propagation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Marker base documenting the (x, coords, dist, kpm, mask) -> h contract."""

    def forward(self, x, coords, dist, key_padding_mask=None, mask=None):
        raise NotImplementedError


def knn_adjacency(
    dist: torch.Tensor,                                # (B, P, P)
    k: int,
    key_padding_mask: Optional[torch.Tensor] = None,   # (B, P) True == PAD
    include_self: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Spatial-kNN adjacency ``A (B, P, P)``: ``A[b,i,j]>0`` iff j is one of i's
    ``k`` nearest non-pad neighbours. With ``normalize=True`` rows sum to 1
    (so ``A @ X`` = mean over each cell's kNN); PAD-query rows are all-zero."""
    B, P, _ = dist.shape
    d = dist.clone()
    big = torch.finfo(d.dtype).max
    if key_padding_mask is not None:
        d = d.masked_fill(key_padding_mask[:, None, :], big)   # don't pick a pad neighbour
    if not include_self:
        eye = torch.eye(P, device=d.device, dtype=torch.bool)
        d = d.masked_fill(eye.unsqueeze(0), big)
    k_eff = max(1, min(int(k), P - (0 if include_self else 1)))
    idx = d.topk(k_eff, dim=-1, largest=False).indices         # (B, P, k_eff)
    A = torch.zeros(B, P, P, device=dist.device, dtype=torch.float32)
    A.scatter_(-1, idx, 1.0)
    if key_padding_mask is not None:
        A = A.masked_fill(key_padding_mask[:, :, None], 0.0)    # pad query -> empty row
        A = A.masked_fill(key_padding_mask[:, None, :], 0.0)    # safety: drop pad neighbours
    if normalize:
        deg = A.sum(-1, keepdim=True).clamp_min(1.0)
        A = A / deg
    return A


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a scalar timestep / mask-fraction ``t (B,)`` ->
    ``(B, dim)``. Used by the diffusion backbone to condition on the noise
    level (== fraction of cells currently masked)."""
    half = max(dim // 2, 1)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float()
        / max(half - 1, 1)
    )
    args = t.reshape(-1).float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if emb.shape[-1] < dim:                                     # pad odd dim
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], dim - emb.shape[-1])], -1)
    return emb[:, :dim]
