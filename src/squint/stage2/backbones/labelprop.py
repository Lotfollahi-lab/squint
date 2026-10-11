"""APPNP-style feature propagation + a light learned correction (Backbone API).

The most parameter-light stage-2 body. It tests how much of the in-painting task
is pure spatial smoothness: the bulk of the work is a fixed, non-parametric
propagation of the input embeddings over the spatial-kNN graph, with only a small
correction MLP that actually learns. This is the "Correct-and-Smooth" idea
(Huang et al. 2020, "Combining Label Propagation and Simple Models...") realised
in feature space, using APPNP / personalised-PageRank propagation
(Klicpera et al. 2019, "Predict then Propagate").

A row-normalised spatial-kNN adjacency ``A (B, P, P)`` is built once per forward
from the dense distance matrix (``A @ h`` = mean over each cell's k nearest
non-pad neighbours). Starting from ``x0 = Linear(x)`` we run APPNP iterations::

    h = x0
    for _ in range(prop_steps):
        h = (1 - alpha) * (A @ h) + alpha * x0      # personalised PageRank

which diffuses observed cells' embeddings into held-out cells along the spatial
graph (``alpha`` = APPNP teleport probability, the weight kept on ``x0``). A
single small correction MLP then refines the smoothed states::

    h = h + MLP(LayerNorm(h))     # MLP = Linear -> GELU -> Dropout -> Linear

Shape trace (B patches, P cells, D = d_model):
    x      (B, P, D) -> x0 (B, P, D)
    A      (B, P, P)
    A @ h  (B, P, D)           # propagation, repeated prop_steps times
    out    (B, P, D)           # same P and cell ordering as the input.

PAD cells get an empty (all-zero) adjacency row from ``knn_adjacency``, so their
neighbour aggregate is zero and they relax toward ``alpha * x0`` — fine, the
parent ignores them via ``key_padding_mask``. The ``coords`` and ``mask`` args
are accepted but unused (held-out cells already carry the [MASK] token in ``x``).
No final LayerNorm or head is applied here (the parent does that).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Stage2Config
from .base import Backbone, knn_adjacency


class LabelPropBackbone(Backbone):
    """APPNP propagation of code embeddings + a small correction MLP."""

    def __init__(self, cfg: Stage2Config) -> None:
        super().__init__()
        mc = cfg.model
        d = mc.d_model
        self.k = mc.graph_knn
        self.prop_steps = mc.prop_steps
        self.alpha = mc.prop_alpha

        # Minimal input projection (in == out == D); the propagation is fixed,
        # so this and the correction MLP below are the only learned parameters.
        self.input_proj = nn.Linear(d, d)

        # Small post-smoothing correction (the "Correct" step), residual.
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(mc.dropout),
            nn.Linear(d, d),
        )

    def forward(self, x, coords, dist, key_padding_mask=None, mask=None):
        # Row-normalised spatial-kNN adjacency; A @ h = per-cell neighbour mean.
        A = knn_adjacency(dist, self.k, key_padding_mask)   # (B, P, P)

        # APPNP / personalised PageRank: diffuse x0 over the spatial graph,
        # teleporting back to x0 with probability alpha at every step.
        x0 = self.input_proj(x)                             # (B, P, D)
        h = x0
        for _ in range(self.prop_steps):
            h = (1.0 - self.alpha) * (A @ h) + self.alpha * x0

        # Light learned correction (residual).
        h = h + self.mlp(self.norm(h))                      # (B, P, D)
        return h
