"""Spatial-kNN message-passing backbone (GraphSAGE-style) behind the Backbone API.

The "predict my codes from my neighbours' codes" body. A row-normalised
spatial-kNN adjacency ``A (B, P, P)`` is built once per forward from the dense
distance matrix (``A @ h`` = mean of each cell's k nearest non-pad neighbours);
``cfg.model.n_layers`` GraphSAGE update layers then propagate over it.

Each layer (post-norm GraphSAGE):

    m = A @ h                          # neighbour aggregate     (B, P, D)
    u = Dropout(GELU(Linear([h, m])))  # concat self + neighbour (in=2D, out=D)
    h = LayerNorm(h + u)               # residual + norm

Shape trace (B patches, P cells, D=d_model):
    x      (B, P, D) -> h
    A      (B, P, P)
    A @ h  (B, P, D)
    cat    (B, P, 2D) -> Linear -> (B, P, D)
    out    (B, P, D)   same P and cell ordering as the input.

PAD cells get an empty (all-zero) adjacency row from ``knn_adjacency``, so their
neighbour aggregate is zero and they simply carry their input forward — fine,
the parent ignores them via ``key_padding_mask``. No final LayerNorm or head is
applied here (the parent ``SpatialCodeTransformer`` does that).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Stage2Config
from .base import Backbone, knn_adjacency


class _SAGELayer(nn.Module):
    """One GraphSAGE update: concat(self, neighbour-aggregate) -> GELU -> residual + norm."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.lin = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        m = A @ h                                  # (B, P, D) neighbour aggregate
        u = self.lin(torch.cat([h, m], dim=-1))    # (B, P, D)
        u = self.dropout(F.gelu(u))
        return self.norm(h + u)                    # residual + post-norm


class GNNBackbone(Backbone):
    """Spatial-kNN message passing (GraphSAGE-style) over the patch graph."""

    def __init__(self, cfg: Stage2Config) -> None:
        super().__init__()
        mc = cfg.model
        self.k = mc.graph_knn
        self.aggregator = mc.gnn_aggregator
        self.layers = nn.ModuleList(
            [_SAGELayer(mc.d_model, mc.dropout) for _ in range(mc.n_layers)]
        )

    def forward(self, x, coords, dist, key_padding_mask=None, mask=None):
        # Build the spatial-kNN adjacency once and reuse it across layers.
        # "mean" -> row-normalised A (A @ h is the per-cell neighbour mean).
        # "sum"  -> un-normalised A  (A @ h is the per-cell neighbour sum).
        # "max"  -> a dense (B, P, P, D) expand would OOM at P~1k, so fall back to mean.
        normalize = self.aggregator != "sum"
        A = knn_adjacency(dist, self.k, key_padding_mask, normalize=normalize)
        h = x
        for layer in self.layers:
            h = layer(h, A)
        return h
