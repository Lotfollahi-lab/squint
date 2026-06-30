"""Transformer backbone (the original stage-2 body) behind the Backbone API.

Distance-biased full self-attention within a patch. Numerically identical to the
pre-refactor model: the parent used to hold ``self.blocks`` directly and loop
them in ``encode``; that loop now lives here.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import Stage2Config
from ..transformer import TransformerBlock
from .base import Backbone


class TransformerBackbone(Backbone):
    def __init__(self, cfg: Stage2Config):
        super().__init__()
        mc = cfg.model
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

    def forward(self, x, coords, dist, key_padding_mask=None, mask=None):
        for blk in self.blocks:
            x = blk(x, dist, key_padding_mask)
        return x
