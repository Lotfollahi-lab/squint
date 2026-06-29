"""
SpatialCodeTransformer -- the stage-2 graph-native Contextual RQ-Transformer.

A bidirectional (MaskGIT-style) transformer over a spatial patch of cells.
Observed cells are embedded from their SQUINT code stacks; held-out cells get a
learnable [MASK] token. After L graph-biased transformer layers, a set of
output heads predicts every (branch, level) code, coarse-to-fine, with
teacher-forced conditioning on earlier predictions:

    targets (default) = [(cell,0), (cell,1), (niche,0), (niche,1)]

    * residual depth  : (b, l) conditions on (b, l') for l' < l  (RQ coarse->fine)
    * hierarchy       : niche levels condition on the cell stack  (mirrors
                        SQUINT's FiLM cell->niche coupling)

Conditioning uses ground-truth codes during training (teacher forcing) and the
decoder's committed codes during inference. Each code-embedding table reserves a
final "unknown" row (index K) used when an earlier target has not been decided
yet (inference only).

Loss is cross-entropy on MASKED cells only, summed over targets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Stage2Config
from .positional import FourierPositionalEncoding
from .transformer import TransformerBlock, pairwise_distances


TargetKey = Tuple[str, int]


def _tkey(branch: str, level: int) -> str:
    return f"{branch}__{level}"


class SpatialCodeTransformer(nn.Module):
    def __init__(self, cfg: Stage2Config):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        d = mc.d_model

        # ----- targets & per-target vocabulary --------------------------------
        self.targets: List[TargetKey] = cfg.prediction_targets
        self.target_K: Dict[str, int] = {}
        for (b, l) in self.targets:
            self.target_K[_tkey(b, l)] = cfg.branch(b).codebook_sizes[l]

        # ----- code embeddings (size K+1; last row = "unknown") ---------------
        # Used both to embed observed input cells and to condition heads.
        self.code_embed = nn.ModuleDict(
            {
                _tkey(b, l): nn.Embedding(self.target_K[_tkey(b, l)] + 1, d)
                for (b, l) in self.targets
            }
        )
        # Optionally a separate set of conditioning embeddings.
        if cfg.model.tie_code_embeddings:
            self.cond_embed = self.code_embed
        else:
            self.cond_embed = nn.ModuleDict(
                {
                    _tkey(b, l): nn.Embedding(self.target_K[_tkey(b, l)] + 1, d)
                    for (b, l) in self.targets
                }
            )

        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)

        # ----- positional encoding -------------------------------------------
        self.pos_enc = FourierPositionalEncoding(
            d_model=d,
            num_freqs=mc.pos_num_freqs,
            sigma=mc.pos_sigma,
            learnable=mc.pos_learnable_freqs,
        )
        self.input_norm = nn.LayerNorm(d)
        self.input_drop = nn.Dropout(mc.dropout)

        # ----- transformer body ----------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d,
                    n_heads=mc.n_heads,
                    d_ff=mc.d_ff,
                    dropout=mc.dropout,
                    dist_bias=mc.attn_dist_bias,
                    gamma_init=mc.attn_gamma_init,
                )
                for _ in range(mc.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

        # ----- per-target conditioning norms + heads -------------------------
        self.ctx_norm = nn.ModuleDict(
            {_tkey(b, l): nn.LayerNorm(d) for (b, l) in self.targets}
        )
        self.heads = nn.ModuleDict(
            {
                _tkey(b, l): nn.Linear(d, self.target_K[_tkey(b, l)])
                for (b, l) in self.targets
            }
        )

        self._init_weights()

    # ------------------------------------------------------------------ utils
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def _allowed(self, ti: int, tj: int) -> bool:
        """May target ti's head condition on (already-decided) target tj?"""
        if tj >= ti:
            return False
        if self.cfg.model.hierarchical:
            return True                         # full chain cell -> niche
        return self.targets[tj][0] == self.targets[ti][0]   # same branch only

    # ----------------------------------------------------------- input embed
    def _embed_inputs(
        self,
        codes: torch.Tensor,                    # (B, P, T) int
        coords: torch.Tensor,                   # (B, P, 2)
        mask: torch.Tensor,                     # (B, P) bool, True == held out
    ) -> torch.Tensor:
        B, P, T = codes.shape
        obs = torch.zeros(B, P, self.cfg.model.d_model, device=codes.device)
        for ti, (b, l) in enumerate(self.targets):
            K = self.target_K[_tkey(b, l)]
            idx = codes[:, :, ti].clamp(0, K - 1)       # observed cells: valid
            obs = obs + self.code_embed[_tkey(b, l)](idx)
        token = torch.where(mask.unsqueeze(-1), self.mask_token.to(obs.dtype), obs)
        token = token + self.pos_enc(coords)
        return self.input_drop(self.input_norm(token))

    # --------------------------------------------------------------- encode
    def encode(
        self,
        codes: torch.Tensor,                    # (B, P, T) int (observed truth)
        coords: torch.Tensor,                   # (B, P, 2) normalised
        mask: torch.Tensor,                     # (B, P) bool, True == held out
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True == PAD
    ) -> torch.Tensor:
        """Transformer body. Returns per-cell hidden states h (B, P, D).

        Note: ``h`` depends only on the observed-cell codes, the mask and the
        positions -- NOT on ``cond_codes``. The decoder exploits this by
        computing ``h`` once per MaskGIT step and re-using it across the
        coarse->fine head evaluations.
        """
        x = self._embed_inputs(codes, coords, mask)
        dist = pairwise_distances(coords)
        for blk in self.blocks:
            x = blk(x, dist, key_padding_mask)
        return self.final_norm(x)

    def head_logits(
        self,
        h: torch.Tensor,                        # (B, P, D)
        ti: int,                                # target index
        cond_codes: torch.Tensor,               # (B, P, T)
    ) -> torch.Tensor:
        """Logits for target ``ti`` given hidden states and conditioning codes."""
        b, l = self.targets[ti]
        ctx = h
        for tj, (bj, lj) in enumerate(self.targets):
            if not self._allowed(ti, tj):
                continue
            Kj = self.target_K[_tkey(bj, lj)]
            cidx = cond_codes[:, :, tj].clamp(0, Kj)        # Kj == "unknown" row
            ctx = ctx + self.cond_embed[_tkey(bj, lj)](cidx)
        ctx = self.ctx_norm[_tkey(b, l)](ctx)
        return self.heads[_tkey(b, l)](ctx)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        codes: torch.Tensor,                    # (B, P, T) int (observed truth)
        coords: torch.Tensor,                   # (B, P, 2) normalised
        mask: torch.Tensor,                     # (B, P) bool, True == held out
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True == PAD
        cond_codes: Optional[torch.Tensor] = None,        # (B, P, T) for heads
    ) -> Dict[str, torch.Tensor]:
        """Returns {target_key: logits (B, P, K_target)}."""
        if cond_codes is None:
            cond_codes = codes                  # teacher forcing
        h = self.encode(codes, coords, mask, key_padding_mask)
        logits: Dict[str, torch.Tensor] = {}
        for ti, (b, l) in enumerate(self.targets):
            logits[_tkey(b, l)] = self.head_logits(h, ti, cond_codes)
        return logits

    # ------------------------------------------------------------------- loss
    def loss(
        self,
        codes: torch.Tensor,                    # (B, P, T) ground truth
        coords: torch.Tensor,
        mask: torch.Tensor,                     # (B, P) True == held out (supervised)
        key_padding_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Masked cross-entropy summed over targets (+ per-target accuracy).

        Only real held-out cells contribute: supervised = mask & ~pad.
        """
        logits = self.forward(
            codes, coords, mask, key_padding_mask, cond_codes=codes
        )
        if key_padding_mask is not None:
            supervised = mask & (~key_padding_mask)
        else:
            supervised = mask
        flat_sup = supervised.reshape(-1)
        n_sup = int(flat_sup.sum().item()) if flat_sup.numel() else 0

        out: Dict[str, torch.Tensor] = {}
        total = codes.new_zeros((), dtype=torch.float32)
        device = codes.device
        for ti, (b, l) in enumerate(self.targets):
            key = _tkey(b, l)
            K = self.target_K[key]
            lg = logits[key].reshape(-1, K)[flat_sup]       # (n_sup, K)
            tg = codes[:, :, ti].reshape(-1)[flat_sup]       # (n_sup,)
            if n_sup == 0:
                ce = torch.zeros((), device=device)
                acc = torch.zeros((), device=device)
            else:
                ce = F.cross_entropy(lg, tg, label_smoothing=label_smoothing)
                acc = (lg.argmax(-1) == tg).float().mean()
            out[f"ce_{key}"] = ce
            out[f"acc_{key}"] = acc
            total = total + ce
        out["loss"] = total
        out["n_supervised"] = torch.tensor(float(n_sup), device=device)
        return out

    # --------------------------------------------------------------- helpers
    @property
    def unknown_index(self) -> Dict[str, int]:
        """The 'unknown' conditioning row index per target (== K)."""
        return dict(self.target_K)
