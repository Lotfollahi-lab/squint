"""
2D positional encoding for irregular cell coordinates (torch).

MaskGIT / RQ-Transformer use learned positional embeddings over a regular image
grid. SQUINT cells live at arbitrary 2D positions, so we use Random Fourier
Features (RFF): a fixed Gaussian frequency matrix B in R^{2 x F} maps a
(normalised) position x in R^2 to

    gamma(x) = [ sin(2*pi * x B) , cos(2*pi * x B) ]  in R^{2F}

then a linear projection to ``d_model``. RFF approximates a shift-invariant
(Gaussian) kernel, so nearby cells get similar encodings and the spectrum is
controlled by ``sigma`` -- the natural continuous analogue of grid position
embeddings. Coordinates are normalised per-patch upstream (see data.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_freqs: int = 64,
        sigma: float = 1.0,
        learnable: bool = False,
        in_dim: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_freqs = num_freqs
        # B: (in_dim, num_freqs) Gaussian frequencies. Fixed buffer by default
        # (reproducible); optionally a learnable parameter.
        B = torch.randn(in_dim, num_freqs) * sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.proj = nn.Linear(2 * num_freqs, d_model)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., in_dim) normalised positions -> (..., d_model)."""
        proj = 2.0 * math.pi * coords @ self.B           # (..., F)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., 2F)
        return self.proj(feats)
