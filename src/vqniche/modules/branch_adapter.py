"""
Lightweight per-branch ADAPTER on a shared encoder trunk.

Parameter-efficient coupling: instead of two full MLP trunks (the decoupled
encoder, ~2x the params) or one shared trunk with no per-branch specialization
(the coupled encoder, which loses resolution), use ONE shared trunk plus a small
adapter per branch. The shared trunk carries the expensive input layers; each
adapter is a tiny dim-preserving map that lets the cell and niche branches
specialize. Every adapter is initialized to (near-)IDENTITY, so at the start of
training the model is exactly the coupled shared-trunk model and the adapters
only learn what specialization is needed.

Three kinds, spanning the cost spectrum:
  - 'affine' : per-feature scale + shift (gamma * z + beta). ~2*dim params.
  - 'lora'   : low-rank residual delta z + B(A z), rank r (B init 0 -> identity).
               ~2*r*dim params.
  - 'mlp'    : a small residual MLP head z + W2(act(W1 z)) (W2 init 0 ->
               identity). ~2*dim*hidden params.

All preserve the feature dim, so codebook embedding dims and decoder shapes are
unchanged relative to the coupled/decoupled encoders.
"""

import torch
import torch.nn as nn


class BranchAdapter(nn.Module):
    def __init__(
            self,
            dim: int,
            kind: str = "lora",
            rank: int = 16,
            hidden: int = None,
            act: str = "gelu",
        ):
        super().__init__()
        self.dim = int(dim)
        self.kind = kind
        if kind == "affine":
            self.gamma = nn.Parameter(torch.ones(self.dim))
            self.beta = nn.Parameter(torch.zeros(self.dim))
        elif kind == "lora":
            r = int(rank)
            self.A = nn.Linear(self.dim, r, bias=False)
            self.B = nn.Linear(r, self.dim, bias=False)
            # B init 0 -> adapter starts as identity (pure residual).
            nn.init.zeros_(self.B.weight)
        elif kind == "mlp":
            h = int(hidden) if hidden else self.dim
            self.fc1 = nn.Linear(self.dim, h)
            self.fc2 = nn.Linear(h, self.dim)
            self.act = nn.GELU() if act == "gelu" else nn.ReLU()
            # fc2 init 0 -> adapter starts as identity (pure residual).
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError(
                f"BranchAdapter kind must be 'affine' | 'lora' | 'mlp', got {kind!r}."
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "affine":
            return self.gamma * z + self.beta
        if self.kind == "lora":
            return z + self.B(self.A(z))
        return z + self.fc2(self.act(self.fc1(z)))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kind={self.kind}"
