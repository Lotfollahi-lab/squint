"""
MMD-based batch-invariance loss for VQNiche_Dual.

Provides a non-adversarial alternative (or complement) to
`adversarial_batch_loss` for pushing the cell-token / latent toward
a batch-invariant distribution. The adversarial CE loss has known
training-dynamic pathologies — in particular, when paired with a
warmup phase the cell-token learns batch-correlated features early,
and the adversary's late-arriving gradient struggles to remove them.
MMD has no min-max game: it computes a kernel-based distance between
the per-batch empirical distributions of the embedding and adds it
directly to the loss. The encoder is pushed to minimise this distance
through ordinary backprop, with no warmup pathology.

Recipe (mirrors what `mmd_comparable` does in the inference-metrics
helper, kept consistent so train-time integration pressure and
test-time integration measurement use the same kernel):
  1. StandardScaler-equivalent normalisation of the latent (zero
     mean, unit variance per dim) on a sub-sampled mini-batch.
  2. RBF kernel with sigma chosen by the median-pairwise-distance
     heuristic on the same sub-sample.
  3. MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 * E[k(x,y)] across the two
     largest batches in the mini-batch (binary-batch case typical
     for mmb-smb; multi-batch is summed pairwise).

Numerical notes:
  - Sub-samples to keep memory / compute under control. With latent
    dim 256 and ~8000 cells per minibatch the full pairwise kernel
    is 256 GB; sampling 512 cells per batch keeps it at ~1 MB.
  - Median-pairwise-distance sigma is computed once per call from
    the SAME sub-sample as the kernel — slightly noisier per-step
    than a corpus-level sigma but tracks scale shifts during
    training.
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


def mmd_batch_loss(
        mmd_target: torch.Tensor,            # (B, d) — embedding to align across batches
        mmd_target_labels: torch.Tensor,     # (B,)   — long-tensor per-cell batch IDs
        wt_mmd_batch: float = 1.0,
        n_sub: int = 512,
    ) -> torch.Tensor:
    """
    Differentiable MMD between per-batch embedding distributions.

    Argument names are deliberately `mmd_target` and `mmd_target_labels`
    (not `z_target` / `batch_labels`) to match the loss_data dict keys
    populated in `VQNiche_Dual._step` — the dispatcher in
    `BaseModel.criterion` passes loss_data values as kwargs whose names
    must match the data dict keys exactly. Using the `mmd_target_*`
    namespace also avoids a name clash with `batch_labels` (which the
    adversarial CE consumes simultaneously when both losses are
    active, with potentially different label slicing).

    Sums pairwise MMD^2 across all (batch_i, batch_j) combinations
    present in the mini-batch (typically only 2 in mmb-smb training).
    Sub-samples up to `n_sub` cells per batch to bound memory.

    Returns the weighted scalar loss `wt_mmd_batch * MMD^2_sum`.

    Edge cases:
      - If only one batch is represented in the mini-batch (rare with
        random shuffling but can happen with small batch_size), MMD is
        undefined; we return a 0 with grad to keep the loss graph alive.
      - If a batch has < 2 cells in the mini-batch, the per-batch
        kernel diagonal is degenerate; we fall back to 0 for that
        pair.
    """
    if mmd_target.numel() == 0:
        return mmd_target.sum() * 0.0   # graph-preserving zero

    device = mmd_target.device

    # 1. Per-dim z-score normalisation on the FULL mini-batch (so the
    #    sigma estimate is on a comparable scale across runs).
    mean = mmd_target.mean(dim=0, keepdim=True)
    std = mmd_target.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (mmd_target - mean) / std

    # 2. Pick the unique batch IDs present in this mini-batch.
    unique = torch.unique(mmd_target_labels)
    if unique.numel() < 2:
        return mmd_target.sum() * 0.0

    # 3. Build per-batch sub-samples (capped at n_sub each).
    samples_per_batch = []
    for b in unique:
        mask = (mmd_target_labels == b)
        n = int(mask.sum().item())
        if n < 2:
            samples_per_batch.append(None)
            continue
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if n > n_sub:
            # Random sub-sample (deterministic per-call via the
            # default torch RNG state, which the trainer seeds).
            sel = idx[torch.randperm(n, device=device)[:n_sub]]
        else:
            sel = idx
        samples_per_batch.append(x[sel])

    valid = [s for s in samples_per_batch if s is not None]
    if len(valid) < 2:
        return mmd_target.sum() * 0.0

    # 4. Estimate sigma via the median-pairwise-distance heuristic on
    #    a pooled sub-sample of the per-batch sub-samples. Cap at
    #    n_sub total points to keep the cdist tractable.
    pooled = torch.cat(valid, dim=0)
    if pooled.shape[0] > n_sub:
        pooled = pooled[torch.randperm(pooled.shape[0], device=device)[:n_sub]]
    pairwise = torch.cdist(pooled, pooled, p=2.0)
    # Drop the zero diagonal before taking the median.
    upper = pairwise[torch.triu_indices(
        pairwise.shape[0], pairwise.shape[1], offset=1, device=device,
    ).unbind(0)]
    sigma = upper.median().detach()
    sigma = sigma.clamp_min(1e-6)

    # 5. Pairwise MMD^2 across all (batch_i, batch_j) present.
    total = mmd_target.new_zeros(())
    n_pairs = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a = valid[i]; b = valid[j]
            k_aa = _rbf_kernel(a, a, sigma)
            k_bb = _rbf_kernel(b, b, sigma)
            k_ab = _rbf_kernel(a, b, sigma)
            mmd_sq = k_aa.mean() + k_bb.mean() - 2.0 * k_ab.mean()
            total = total + mmd_sq
            n_pairs += 1

    if n_pairs == 0:
        return mmd_target.sum() * 0.0
    return total / n_pairs * float(wt_mmd_batch)
