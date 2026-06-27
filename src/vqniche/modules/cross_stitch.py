"""
Cross-stitch unit (Misra et al., CVPR 2016) for two encoder branches.

A cross-stitch unit learns a 2x2 linear combination of two same-shaped
activation tensors — here the cell-branch post-MLP latent and the
niche-branch post-MLP latent — and returns the two mixed tensors:

    z_cell'  = a_cc * z_cell + a_cn * z_niche
    z_niche' = a_nc * z_cell + a_nn * z_niche

The mixed cell latent then feeds VQ_cell, and the mixed niche latent
feeds the GNN -> VQ_niche. A single cross-stitch network spans the
entire shared-to-split spectrum and *learns* the optimal amount of
sharing automatically, removing the need to hand-pick a split point
(Misra et al. 2016). It does NOT physically merge a trunk, so it keeps
the two branches' full capacity (roughly doubles the encoder MLP params,
same as the decoupled architecture it builds on).

Parameters
----------
dim : int
    Feature dimension of each branch's latent (cell and niche MLP
    output dims must be equal).
per_channel : bool, default False
    False -> ONE 2x2 mixing matrix shared across all `dim` channels
    (4 learnable scalars, maximally parsimonious — the "elegant"
    paper variant). True -> an independent 2x2 per channel
    (4 * dim scalars), the per-unit form closest to the original
    paper's channel-wise alphas.
init_diag : float, default 0.9
    Initial value of the same-branch coefficients a_cc, a_nn.
init_off : float, default 0.1
    Initial value of the cross-branch coefficients a_cn, a_nc.
    Near-identity init (0.9 / 0.1) means each branch mostly passes
    through its own activations at the start and only a small fraction
    of the other branch leaks in; the optimiser then learns how much
    to share. Rows are NOT renormalised — the coefficients are free.

Notes
-----
Vanilla cross-stitch (this module) mixes matched channels only; NDDR-CNN
(Gao et al. 2018) fuses across all channels via a 1x1 conv. For a small
MLP encoder the matched-channel 2x2 is the standard, lightweight choice.
"""

import torch
import torch.nn as nn


class CrossStitch(nn.Module):
    def __init__(
            self,
            dim: int,
            per_channel: bool = False,
            init_diag: float = 0.9,
            init_off: float = 0.1,
        ):
        super().__init__()
        self.dim = int(dim)
        self.per_channel = bool(per_channel)
        shape = (self.dim,) if self.per_channel else (1,)
        self.a_cc = nn.Parameter(torch.full(shape, float(init_diag)))
        self.a_cn = nn.Parameter(torch.full(shape, float(init_off)))
        self.a_nc = nn.Parameter(torch.full(shape, float(init_off)))
        self.a_nn = nn.Parameter(torch.full(shape, float(init_diag)))

    def forward(self, z_cell: torch.Tensor, z_niche: torch.Tensor):
        """
        Parameters
        ----------
        z_cell, z_niche : (N, dim)
            Per-branch latents (same shape).

        Returns
        -------
        (mixed_cell, mixed_niche) : both (N, dim)
        """
        # Coefficients are (1,) or (dim,) and broadcast over the batch
        # dimension of the (N, dim) latents.
        mixed_cell = self.a_cc * z_cell + self.a_cn * z_niche
        mixed_niche = self.a_nc * z_cell + self.a_nn * z_niche
        return mixed_cell, mixed_niche

    def extra_repr(self) -> str:
        return f"dim={self.dim}, per_channel={self.per_channel}"
