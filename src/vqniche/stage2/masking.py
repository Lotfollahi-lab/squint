"""
Masking strategies for stage-2 training (pure numpy).

The stage-2 model is trained by masking some cells' code stacks and predicting
them from the rest. Two strategies:

* ``block``  -- a spatially CONTIGUOUS region is held out. This matches the
  region-holdout geometry the model is evaluated on (in-painting a hole in the
  tissue). Training with contiguous blocks is important: a model trained only
  on random-scatter masks learns to interpolate from immediate neighbours and
  fails on contiguous holes where no masked cell has an observed neighbour.

* ``random`` -- independent Bernoulli masking (a MaskGIT-style baseline /
  ablation; good for learning local code statistics, weak for large holes).

All functions are deterministic given a ``numpy.random.Generator``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def random_mask(
    n: int,
    frac: float,
    rng: np.random.Generator,
    min_masked: int = 1,
) -> np.ndarray:
    """Independent Bernoulli mask. Returns bool array, True == held out."""
    n_mask = max(int(min_masked), int(round(frac * n)))
    n_mask = min(n_mask, n - 1)         # always keep >=1 observed cell
    mask = np.zeros(n, dtype=bool)
    if n_mask <= 0:
        return mask
    idx = rng.choice(n, size=n_mask, replace=False)
    mask[idx] = True
    return mask


def block_mask(
    coords: np.ndarray,
    frac: float,
    rng: np.random.Generator,
    seed_idx: Optional[int] = None,
    min_masked: int = 1,
) -> np.ndarray:
    """Contiguous spatial block: hold out the ``frac`` cells closest to a seed.

    Picking the ``k`` nearest cells to a random seed carves out a spatial disk
    -- a connected hole -- which is exactly the in-painting test geometry.

    Parameters
    ----------
    coords : (n, 2) float array of cell positions (any units).
    frac   : fraction of the ``n`` cells to mask.
    seed_idx : optionally fix the disk centre (else chosen uniformly at random).

    Returns
    -------
    (n,) bool array, True == held out. The seed cell is always masked.
    """
    n = coords.shape[0]
    n_mask = max(int(min_masked), int(round(frac * n)))
    n_mask = min(n_mask, n - 1)
    mask = np.zeros(n, dtype=bool)
    if n_mask <= 0:
        return mask
    if seed_idx is None:
        seed_idx = int(rng.integers(n))
    d2 = np.sum((coords - coords[seed_idx]) ** 2, axis=1)
    nearest = np.argsort(d2, kind="stable")[:n_mask]
    mask[nearest] = True
    return mask


def make_mask(
    coords: np.ndarray,
    kind: str,
    frac_min: float,
    frac_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a mask of the configured kind with a random held-out fraction."""
    frac = float(rng.uniform(frac_min, frac_max))
    if kind == "block":
        return block_mask(coords, frac, rng)
    if kind == "random":
        return random_mask(coords.shape[0], frac, rng)
    raise ValueError(f"unknown mask kind {kind!r}")
