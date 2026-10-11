"""
Disentanglement penalty between the cell and niche latents.

Idea #2 for making a coupled architecture beat the decoupled baseline: if the
cell-intrinsic code and the spatial-niche code are partly REDUNDANT (both
encoding cell-type), force them apart so the niche code is pushed to carry only
the COMPLEMENTARY (spatial / neighbourhood) signal. This is "coupling" via a
joint constraint that makes the two codes work together by being complementary,
rather than via parameter sharing.

Mechanism (Barlow-Twins-style cross-correlation decorrelation; Zbontar et al.
2021): batch-normalise each latent per feature (zero mean, unit variance across
the batch), form the cross-correlation matrix between the cell and niche
features, and penalise its squared entries. Driving the cross-correlation to
zero makes every cell feature linearly uncorrelated with every niche feature.

Unlike the within-branch Barlow-Twins redundancy term, here the two views are
the TWO BRANCHES of one cell (cell latent vs niche latent), and there is no
invariance term — we only want to remove shared linear information, not align
them.
"""

import torch


def disentangle_cell_niche_loss(
        z_cell: torch.Tensor,
        z_niche: torch.Tensor,
        wt_disentangle: float = 1.0,
        eps: float = 1e-5,
        **kwargs,
    ) -> torch.Tensor:
    """
    Parameters
    ----------
    z_cell : (N, d_cell)
        Cell-branch latent for the N seed cells (e.g. z_mlp[:batch_size]).
    z_niche : (N, d_niche)
        Niche-branch latent for the same N seed cells (e.g. z_gnn[:batch_size]).
    wt_disentangle : float
        Loss weight.
    eps : float
        Numerical floor on the per-feature std used for normalisation.

    Returns
    -------
    scalar tensor = wt_disentangle * mean(cross_correlation ** 2).

    The mean (rather than sum) over the d_cell x d_niche entries keeps the
    penalty magnitude scale-free (roughly the mean squared linear correlation,
    in [0, 1]) so the weight transfers across codebook/embedding dims.
    """
    n = z_cell.shape[0]
    if n < 2:
        # Cross-correlation is undefined for <2 samples; return a grad-safe 0.
        return z_cell.sum() * 0.0

    zc = (z_cell - z_cell.mean(dim=0, keepdim=True)) / (
        z_cell.std(dim=0, keepdim=True) + eps)
    zn = (z_niche - z_niche.mean(dim=0, keepdim=True)) / (
        z_niche.std(dim=0, keepdim=True) + eps)

    # (d_cell, d_niche) cross-correlation between the two branches' features.
    c = (zc.T @ zn) / n
    penalty = (c ** 2).mean()
    return wt_disentangle * penalty


def cell_niche_alignment_loss(
        z_cell: torch.Tensor,
        z_niche: torch.Tensor,
        wt_align: float = 1.0,
        eps: float = 1e-5,
        **kwargs,
    ) -> torch.Tensor:
    """
    Cross-branch ALIGNMENT (the opposite sign of `disentangle_cell_niche_loss`):
    push the matched cell/niche feature dimensions to be CORRELATED — the
    Barlow-Twins INVARIANCE term. Tests whether *aligning* the two codes helps
    where decorrelating them hurt. Batch-normalise each latent per feature,
    take the diagonal of the cross-correlation over the first
    m = min(d_cell, d_niche) dims, and penalise its distance from 1:

        wt_align * mean_i (1 - corr(z_cell_i, z_niche_i)) ** 2 .

    Dimension-matched (uses the first m dims of each branch); no extra params.
    """
    n = z_cell.shape[0]
    if n < 2:
        return z_cell.sum() * 0.0
    m = min(z_cell.shape[1], z_niche.shape[1])
    zc = z_cell[:, :m]
    zn = z_niche[:, :m]
    zc = (zc - zc.mean(dim=0, keepdim=True)) / (zc.std(dim=0, keepdim=True) + eps)
    zn = (zn - zn.mean(dim=0, keepdim=True)) / (zn.std(dim=0, keepdim=True) + eps)
    diag_corr = (zc * zn).mean(dim=0)          # (m,) per-feature correlation
    penalty = ((1.0 - diag_corr) ** 2).mean()
    return wt_align * penalty
