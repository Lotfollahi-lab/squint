"""GraphMAE-style masked graph autoencoder backbone behind the Backbone API.

Implements the encoder -> re-mask -> decoder pattern of GraphMAE (Hou et al.,
"GraphMAE: Self-Supervised Masked Graph Autoencoders", KDD 2022,
https://arxiv.org/abs/2205.10803). The held-out cells already carry the parent's
learnable ``[MASK]`` token on input; a GraphSAGE-style GNN encoder propagates
over ALL cells to an encoded representation ``z``, GraphMAE's "fixed re-mask
decoding" then OVERWRITES the encoded vectors of the held-out cells with a
second, decoder-side learnable re-mask token, and a GNN decoder propagates again
to produce the per-cell hidden states ``h``.

Re-masking forces the decoder to reconstruct the held-out cells from the encoded
context of their neighbours rather than from their own (leaky) encoded features,
which is the core trick that makes GraphMAE's feature reconstruction work.

Both encoder and decoder use the same message-passing layer as the ``gnn``
backbone (concat self + spatial-kNN neighbour aggregate -> GELU -> residual +
post-norm), over a single row-normalised spatial-kNN adjacency built once per
forward from the dense distance matrix.

Shape trace (B patches, P cells, D=d_model):
    x      (B, P, D) -> h0
    A      (B, P, P)            spatial-kNN adjacency (PAD rows all-zero)
    encode n_enc layers         -> z   (B, P, D)
    re-mask z[mask] := remask_token   (B, P, D), unchanged P/order
    decode n_dec layers         -> h   (B, P, D)
    out    (B, P, D)            same P and cell ordering as the input.

No final LayerNorm or prediction head is applied here (the parent
``SpatialCodeTransformer`` does that), and P / the cell ordering are preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Stage2Config
from .base import Backbone, knn_adjacency


class _MPLayer(nn.Module):
    """One GraphSAGE-style message-passing update.

        m = A @ h                          # spatial-kNN neighbour aggregate
        u = Dropout(GELU(Linear([h, m])))  # concat self + neighbour (in=2D, out=D)
        h = LayerNorm(h + u)               # residual + post-norm
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.lin = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        m = A @ h                                   # (B, P, D) neighbour aggregate
        u = self.lin(torch.cat([h, m], dim=-1))     # (B, P, D)
        u = self.dropout(F.gelu(u))
        return self.norm(h + u)                     # residual + post-norm


class GraphMAEBackbone(Backbone):
    """GraphMAE-style masked graph autoencoder (encoder -> re-mask -> decoder)."""

    def __init__(self, cfg: Stage2Config) -> None:
        super().__init__()
        mc = cfg.model
        self.k = mc.graph_knn

        # Split the layer budget across the encoder and decoder GNNs; both get
        # at least one layer even for tiny ``n_layers``.
        n_enc = max(1, mc.n_layers // 2)
        n_dec = max(1, mc.n_layers - n_enc)
        self.encoder = nn.ModuleList(
            [_MPLayer(mc.d_model, mc.dropout) for _ in range(n_enc)]
        )
        self.decoder = nn.ModuleList(
            [_MPLayer(mc.d_model, mc.dropout) for _ in range(n_dec)]
        )

        # Decoder-side "fixed re-mask" token (GraphMAE): replaces the encoded
        # vectors of held-out cells before decoding.
        self.remask_token = nn.Parameter(torch.zeros(mc.d_model))
        nn.init.normal_(self.remask_token, std=0.02)

    def forward(self, x, coords, dist, key_padding_mask=None, mask=None):
        # One row-normalised spatial-kNN adjacency, reused by every MP layer.
        # PAD cells get an empty (all-zero) row, so their neighbour aggregate is
        # zero and they simply carry their input forward.
        A = knn_adjacency(dist, self.k, key_padding_mask)

        # ENCODER: message passing over ALL cells -> encoded z (B, P, D).
        z = x
        for layer in self.encoder:
            z = layer(z, A)

        # RE-MASK: overwrite the encoded vectors of held-out cells with the
        # learnable re-mask token (broadcast over the batch / cell dims). Skipped
        # when no held-out mask is supplied (e.g. pure encoding).
        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.remask_token, z)

        # DECODER: more message passing -> per-cell hidden states h (B, P, D).
        h = z
        for layer in self.decoder:
            h = layer(h, A)
        return h
