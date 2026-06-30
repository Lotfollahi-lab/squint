"""GraphGPS backbone (the local-MPNN + global-attention hybrid) behind the Backbone API.

GraphGPS (Rampášek et al., 2022, "Recipe for a General, Powerful, Scalable Graph
Transformer", NeurIPS 2022) runs, in every layer and *in parallel*, a LOCAL
message-passing branch (captures short-range graph structure) and a GLOBAL
attention branch (every cell can attend to every other cell in the patch), then
sums the two branch outputs back into the residual stream before a feed-forward
block.

Here the local branch is a minimal mean-aggregation MPNN over the spatial-kNN
graph (``A @ h`` with ``A`` row-normalised), concatenated with the node's own
features and projected; the global branch reuses SQUINT's
``GraphBiasedSelfAttention`` (distance-biased full self-attention within the
patch). This keeps the GPS backbone a clean head-to-head ablation against the
plain transformer / gnn / labelprop bodies: it shares the same embedding, heads,
loss and MaskGIT decode, and only the body differs.

Shape-trace (with B=2, P=64, D=256):
    x      (2, 64, 256)   ->  h (2, 64, 256)         (P and ordering preserved)
    A = knn_adjacency(dist, k, kpm)  (2, 64, 64)     (row-normalised mean MP)
    per layer (pre-norm GraphGPS combine):
        xn    = norm1(h)                (2, 64, 256)
        local = MPNN([xn, A @ xn])      (2, 64, 256)  (Linear 2D->D, GELU, drop)
        glob  = attn(xn, dist, kpm)     (2, 64, 256)
        h     = h + drop(local) + drop(glob)
        h     = h + drop(ff(norm2(h)))  (2, 64, 256)
    returns h (2, 64, 256); no final norm / heads (parent applies those).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import Stage2Config
from ..transformer import GraphBiasedSelfAttention
from .base import Backbone, knn_adjacency


class GPSLayer(nn.Module):
    """One GraphGPS layer: parallel local MPNN + global attention, then FF.

    Pre-norm, with the two branches read off a shared normalised input and their
    (dropped) outputs added back into the residual stream, mirroring the GraphGPS
    ``X' = X + MPNN(X) + GlobalAttn(X)`` recipe (Rampášek et al., 2022).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        dist_bias: bool = True,
        gamma_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Local message-passing branch: concat([h, A @ h]) -> D.
        self.mpnn = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Global attention branch (distance-biased full self-attention).
        self.attn = GraphBiasedSelfAttention(
            d_model, n_heads, dropout, dist_bias=dist_bias, gamma_init=gamma_init
        )

        # Feed-forward block.
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,                                   # (B, P, D)
        A: torch.Tensor,                                   # (B, P, P) row-normalised kNN adjacency
        dist: Optional[torch.Tensor] = None,               # (B, P, P) pairwise distances
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, P) True == PAD
    ) -> torch.Tensor:
        xn = self.norm1(h)                                 # shared pre-norm input
        local = self.mpnn(torch.cat([xn, A @ xn], dim=-1))  # local MPNN branch
        glob = self.attn(xn, dist, key_padding_mask)        # global attention branch
        h = h + self.drop(local) + self.drop(glob)          # GraphGPS combine
        h = h + self.drop(self.ff(self.norm2(h)))           # feed-forward
        return h


class GPSBackbone(Backbone):
    """GraphGPS stage-2 body: per-layer parallel MPNN + global attention.

    The spatial-kNN adjacency ``A`` is computed once in ``forward`` (it depends
    only on ``dist`` and the padding mask, which are fixed across layers) and
    threaded into every layer, which is cleaner than recomputing it per layer.
    """

    def __init__(self, cfg: Stage2Config) -> None:
        super().__init__()
        mc = cfg.model
        self.graph_knn = mc.graph_knn
        self.layers = nn.ModuleList(
            [
                GPSLayer(
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

    def forward(
        self,
        x: torch.Tensor,                                   # (B, P, D)
        coords: torch.Tensor,                              # (B, P, 2)
        dist: torch.Tensor,                                # (B, P, P)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, P) True == PAD
        mask: Optional[torch.Tensor] = None,               # (B, P) True == held-out (unused)
    ) -> torch.Tensor:
        # Row-normalised spatial-kNN adjacency (mean message passing); shared
        # across layers since dist / padding do not change.
        A = knn_adjacency(dist, self.graph_knn, key_padding_mask)
        h = x
        for layer in self.layers:
            h = layer(h, A, dist, key_padding_mask)
        return h
