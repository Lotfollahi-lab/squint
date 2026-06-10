"""
MMD-to-prior batch-invariance loss for VQNiche_Dual.

Non-adversarial, non-parametric analogue of a VAE's KL(q(z|x) || N(0, I))
divergence. Instead of making the encoder probabilistic (outputting mu,
sigma) and computing closed-form KL, this loss matches the EMPIRICAL
distribution of the embedding directly to samples from an isotropic
Gaussian prior, via a kernel-based MMD distance. Encoder stays
deterministic; no reparameterisation needed.

Mechanism (analogous to NicheCompass's KL-based integration):
  - The prior N(0, prior_std² * I) is batch-agnostic by construction.
  - Pulling the embedding's empirical distribution toward the prior
    forces per-batch distributions to converge to the same target →
    cells from different batches end up in overlapping latent
    regions → integrated.
  - Combined with `+decoder-cov` (which absorbs per-batch differences
    on the decoder side), this mirrors NicheCompass's KL-divergence
    integration mechanism without needing a parametric encoder or
    adversarial min-max game.

Compared to the sibling MMD batch loss (`mmd_batch.py`):
  - `mmd_batch_loss`: matches per-batch empirical distributions to
    EACH OTHER (requires batch labels; pairwise MMD across batches).
  - `mmd_prior_loss` (this file): matches the empirical distribution
    to a FIXED prior (no batch labels needed; single MMD vs prior
    samples).

Both can be combined for a "match-to-each-other + match-to-prior"
multi-axis pressure.

Numerical notes:
  - Same RBF kernel + median-pairwise-distance sigma heuristic as
    the batch-MMD sibling, for consistency.
  - We do NOT z-score normalise `mmd_target` before computing MMD,
    UNLIKE `mmd_batch_loss`. The point of MMD-to-prior is to pull
    the embedding's natural scale + shape toward the prior; if we
    normalised first, mean=0 / std=1 would be trivially satisfied
    and the loss would only measure higher-order moments. Without
    normalisation the encoder is pulled to produce z with both the
    correct scale and the correct distribution shape.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _rbf_kernel(
        a: torch.Tensor,           # (n_a, d)
        b: torch.Tensor,           # (n_b, d)
        sigma: torch.Tensor,       # scalar
    ) -> torch.Tensor:
    """RBF kernel matrix k(a, b) = exp(-||a-b||^2 / (2*sigma^2))."""
    dist_sq = torch.cdist(a, b, p=2.0).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma.pow(2) + 1e-12))


def mmd_prior_loss(
        mmd_target: torch.Tensor,        # (B, d) — embedding to match to prior
        wt_mmd_prior: float = 1.0,
        n_sub: int = 512,
        prior_std: float = 1.0,
    ) -> torch.Tensor:
    """
    Differentiable MMD between `mmd_target` and samples from the
    isotropic Gaussian prior N(0, prior_std² * I).

    Argument name `mmd_target` matches the loss_data dict key
    populated in `VQNiche_Dual._step` (same as `mmd_batch_loss`), so
    both losses can coexist without a duplicate data-key registration
    — they consume the same tensor but compute different distances.
    Batch labels are NOT needed (and not consumed) since the target
    distribution is the prior, not per-batch distributions.

    Sub-samples up to `n_sub` cells from `mmd_target` to bound memory
    on the kernel matrix. Draws an equal-size sample from the prior
    fresh each call (no caching).

    Parameters
    ----------
    mmd_target
        Embedding tensor of shape (B, d). For SQUINT typically
        `z_mlp[:batch_size]` (the cell-token input pre-VQ).
    wt_mmd_prior
        Loss weight. Returned scalar is `wt_mmd_prior * MMD^2`.
    n_sub
        Sub-sample size for the kernel computation. Default 512.
    prior_std
        Standard deviation of the isotropic Gaussian prior. Default
        1.0 (standard N(0, I)). Increase if the encoder's natural
        scale is large and aggressive scale-pulling fights
        reconstruction.

    Edge cases:
      - Empty input: returns a graph-preserving zero.
      - Single-sample input: cannot compute kernel diagonals
        meaningfully; returns 0.
    """
    if mmd_target.numel() == 0:
        return mmd_target.sum() * 0.0   # graph-preserving zero

    device = mmd_target.device
    x = mmd_target

    # Sub-sample to keep the pairwise kernel matrix tractable. Use
    # the entire mini-batch if it's already smaller than n_sub.
    if x.shape[0] > n_sub:
        idx = torch.randperm(x.shape[0], device=device)[:n_sub]
        x = x[idx]

    # Need at least 2 points for the kernel diagonal estimates.
    if x.shape[0] < 2:
        return mmd_target.sum() * 0.0

    # Sample from the prior — same shape as the (possibly sub-sampled)
    # x. The prior is N(0, prior_std² * I) with the same dim as x.
    p = torch.randn_like(x) * float(prior_std)

    # Estimate sigma via the median-pairwise-distance heuristic on
    # the POOLED x + p (so the kernel adapts to whatever scale the
    # encoder is currently producing). Same recipe as the batch-MMD
    # sibling for consistency.
    pooled = torch.cat([x, p], dim=0)
    if pooled.shape[0] > n_sub:
        pooled = pooled[torch.randperm(pooled.shape[0], device=device)[:n_sub]]
    pairwise = torch.cdist(pooled, pooled, p=2.0)
    upper = pairwise[torch.triu_indices(
        pairwise.shape[0], pairwise.shape[1], offset=1, device=device,
    ).unbind(0)]
    sigma = upper.median().detach().clamp_min(1e-6)

    # MMD^2 = E[k(x,x')] + E[k(p,p')] - 2 * E[k(x,p)].
    k_xx = _rbf_kernel(x, x, sigma)
    k_pp = _rbf_kernel(p, p, sigma)
    k_xp = _rbf_kernel(x, p, sigma)
    mmd_sq = k_xx.mean() + k_pp.mean() - 2.0 * k_xp.mean()

    return mmd_sq * float(wt_mmd_prior)
