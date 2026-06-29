"""
Data core for the stage-2 spatial code prior (pure numpy / anndata).

Nothing here imports torch, so the patching / masking / normalisation logic is
unit-testable without the ML stack. The torch ``Dataset`` / ``DataModule`` that
wrap this core live in ``datamodule.py``.

Pipeline
--------
1. ``AnnDataCodeSource`` reads a FROZEN ``predicted_adata.h5ad`` written by
   SQUINT's predict pipeline and exposes, per cell:
       * code stacks per branch  (uns['Indices_cell'], uns['Indices_niche'])
       * 2D position             (obsm['spatial'])
       * tissue section id       (obs['adata_batch_id'])
   It NEVER reads expression.

2. ``PatchSampler`` draws a connected spatial patch (a disk of ``patch_size``
   cells around a random seed, within one section).

3. ``Patch`` carries the per-patch arrays + a held-out mask + per-patch
   coordinate normalisation, ready to be tensorised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import BranchSpec, DataConfig, Stage2Config
from . import masking


# ---------------------------------------------------------------------------
# numpy kNN (sklearn if available, brute-force fallback)
# ---------------------------------------------------------------------------
def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """(n, k) indices of the k nearest neighbours of each row (excludes self).

    Uses sklearn's KDTree when available; falls back to a vectorised
    brute-force computation (fine for patch-sized inputs).
    """
    n = coords.shape[0]
    k = int(min(k, max(1, n - 1)))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        _, idx = nn.kneighbors(coords)
        return idx[:, 1:]               # drop self (column 0)
    except Exception:
        d2 = _pairwise_sq_dists(coords)
        np.fill_diagonal(d2, np.inf)
        return np.argsort(d2, axis=1, kind="stable")[:, :k]


def _pairwise_sq_dists(coords: np.ndarray) -> np.ndarray:
    sq = np.sum(coords ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * coords @ coords.T
    return np.maximum(d2, 0.0)


def median_knn_distance(coords: np.ndarray, k: int = 8) -> float:
    """Median distance to the k-th nearest neighbour -- a robust length scale."""
    n = coords.shape[0]
    if n < 2:
        return 1.0
    idx = knn_indices(coords, min(k, n - 1))
    nn_d = np.sqrt(np.sum((coords[idx[:, -1]] - coords) ** 2, axis=1))
    med = float(np.median(nn_d))
    return med if med > 0 else 1.0


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
class AnnDataCodeSource:
    """Frozen SQUINT codes + positions, grouped by tissue section.

    Parameters
    ----------
    adata : an ``anndata.AnnData`` already in memory, or a path to a
        ``predicted_adata.h5ad``.
    branches : the branch specs to model (defaults to SQUINT's cell + niche).
    """

    def __init__(
        self,
        adata,
        branches: Optional[Sequence[BranchSpec]] = None,
        coord_key: str = "spatial",
        section_key: str = "adata_batch_id",
    ):
        if isinstance(adata, str):
            import anndata

            adata = anndata.read_h5ad(adata)

        self.coord_key = coord_key
        self.section_key = section_key

        if coord_key not in adata.obsm:
            raise KeyError(f"adata.obsm has no '{coord_key}' (spatial coords)")
        self.coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)[:, :2]
        self.n_cells = self.coords.shape[0]

        # branch codes + sizes (read from the frozen adata; resolve sizes)
        if branches is None:
            branches = self._infer_default_branches(adata)
        self.branches: List[BranchSpec] = []
        self.codes: Dict[str, np.ndarray] = {}
        for b in branches:
            idx = self._read_indices(adata, b)
            sizes = self._resolve_sizes(adata, b, idx)
            b = BranchSpec(b.name, sizes, uns_key=b.uns_key, sizes_key=b.sizes_key)
            self.branches.append(b)
            self.codes[b.name] = idx

        # section / batch ids -> contiguous integer labels
        self.section = self._read_sections(adata)
        self.section_ids = np.unique(self.section)
        self._section_rows = {
            int(s): np.where(self.section == s)[0] for s in self.section_ids
        }

    # ---- adata readers ----------------------------------------------------
    @staticmethod
    def _to_numpy(x) -> Optional[np.ndarray]:
        if x is None:
            return None
        if hasattr(x, "detach"):        # torch tensor
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _read_indices(self, adata, b: BranchSpec) -> np.ndarray:
        idx = None
        if b.uns_key in adata.uns:
            idx = self._to_numpy(adata.uns[b.uns_key])
        else:
            obsm_key = f"{b.name}_code_indices"
            obs_key = f"{b.name}_code_index"
            if obsm_key in adata.obsm:
                idx = self._to_numpy(adata.obsm[obsm_key])
            elif obs_key in adata.obs:
                idx = self._to_numpy(adata.obs[obs_key].values)
        if idx is None:
            raise KeyError(
                f"could not find code indices for branch '{b.name}' "
                f"(looked for uns['{b.uns_key}'], obsm['{b.name}_code_indices'], "
                f"obs['{b.name}_code_index'])"
            )
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx[:, None]
        if idx.shape[0] != self.n_cells:
            raise ValueError(
                f"branch '{b.name}': {idx.shape[0]} code rows != "
                f"{self.n_cells} cells"
            )
        return idx.astype(np.int64)

    def _resolve_sizes(self, adata, b: BranchSpec, idx: np.ndarray) -> List[int]:
        sizes = None
        if b.sizes_key in adata.uns:
            sizes = adata.uns[b.sizes_key]
        elif b.name == "niche" and "codebook_sizes" in adata.uns:  # legacy alias
            sizes = adata.uns["codebook_sizes"]
        if sizes is not None:
            sizes = [int(s) for s in np.asarray(sizes).ravel().tolist()]
        if not sizes or len(sizes) != idx.shape[1]:
            sizes = [int(idx[:, q].max()) + 1 for q in range(idx.shape[1])]
        return sizes

    def _read_sections(self, adata) -> np.ndarray:
        if self.section_key in adata.obs:
            raw = adata.obs[self.section_key].values
        elif self.section_key in adata.uns:
            raw = self._to_numpy(adata.uns[self.section_key]).ravel()
        else:
            # single section
            return np.zeros(self.n_cells, dtype=np.int64)
        # map arbitrary labels -> 0..S-1
        _, inv = np.unique(np.asarray(raw), return_inverse=True)
        return inv.astype(np.int64)

    @staticmethod
    def _infer_default_branches(adata) -> List[BranchSpec]:
        from .config import default_branches

        # sizes resolved later; pass placeholders (overwritten in __init__).
        return default_branches([1], [1])

    # ---- convenience ------------------------------------------------------
    def branch_sizes(self) -> Dict[str, List[int]]:
        return {b.name: list(b.codebook_sizes) for b in self.branches}

    def section_of(self, s: int) -> np.ndarray:
        return self._section_rows[int(s)]


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------
@dataclass
class Patch:
    """One training/inference example: a connected spatial patch."""

    global_idx: np.ndarray              # (P,) row indices into the source
    coords_raw: np.ndarray              # (P, 2) original positions
    coords_norm: np.ndarray             # (P, 2) centred + scaled positions
    codes: Dict[str, np.ndarray]        # branch -> (P, L) int codes
    mask: np.ndarray                    # (P,) bool, True == held out
    section: int

    @property
    def size(self) -> int:
        return self.global_idx.shape[0]


def normalise_coords(coords: np.ndarray, mode: str, knn_k: int = 8) -> np.ndarray:
    """Centre at the patch centroid and divide by a length scale."""
    centred = coords - coords.mean(axis=0, keepdims=True)
    if mode == "knn":
        scale = median_knn_distance(coords, k=knn_k)
    else:  # "std"
        scale = float(np.sqrt((centred ** 2).sum(axis=1).mean()))
        scale = scale if scale > 0 else 1.0
    return centred / scale


# ---------------------------------------------------------------------------
# Patch sampler
# ---------------------------------------------------------------------------
class PatchSampler:
    """Draws connected spatial patches and applies a held-out mask."""

    def __init__(self, source: AnnDataCodeSource, cfg: DataConfig):
        self.source = source
        self.cfg = cfg

    # ---- patch geometry ---------------------------------------------------
    def _grow_patch(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        """Pick a section + a disk of patch_size cells around a random seed."""
        # weight sections by their cell count so big sections are sampled more.
        counts = np.array(
            [self.source.section_of(s).size for s in self.source.section_ids],
            dtype=np.float64,
        )
        probs = counts / counts.sum()
        s = int(self.source.section_ids[rng.choice(len(probs), p=probs)])
        rows = self.source.section_of(s)
        coords = self.source.coords[rows]

        p = min(self.cfg.patch_size, rows.size)
        seed_local = int(rng.integers(rows.size))
        d2 = np.sum((coords - coords[seed_local]) ** 2, axis=1)
        order = np.argsort(d2, kind="stable")[:p]
        return s, rows[order]

    def sample(self, rng: np.random.Generator) -> Patch:
        s, gidx = self._grow_patch(rng)
        coords = self.source.coords[gidx]
        coords_n = normalise_coords(coords, self.cfg.coord_norm, self.cfg.knn)
        mask = masking.make_mask(
            coords_n,
            self.cfg.mask_kind,
            self.cfg.mask_frac_min,
            self.cfg.mask_frac_max,
            rng,
        )
        codes = {b.name: self.source.codes[b.name][gidx] for b in self.source.branches}
        return Patch(
            global_idx=gidx,
            coords_raw=coords,
            coords_norm=coords_n,
            codes=codes,
            mask=mask,
            section=s,
        )

    def epoch_len(self) -> int:
        return max(1, int(self.cfg.oversample * self.source.n_cells / self.cfg.patch_size))


# ---------------------------------------------------------------------------
# Inference patch builder (in-painting a known held-out region)
# ---------------------------------------------------------------------------
def inpainting_patch(
    source: AnnDataCodeSource,
    holdout_idx: Sequence[int],
    cfg: DataConfig,
    context_radius_mult: float = 2.0,
) -> Patch:
    """Build a patch around a KNOWN held-out region for in-painting.

    The held-out cells are masked; a surrounding ring of observed cells (within
    ``context_radius_mult`` x the region radius, capped at ``patch_size``) is
    included as context. All held-out cells must share one section.
    """
    holdout_idx = np.asarray(list(holdout_idx), dtype=np.int64)
    secs = np.unique(source.section[holdout_idx])
    if secs.size != 1:
        raise ValueError("all held-out cells must be in the same section")
    s = int(secs[0])
    rows = source.section_of(s)
    coords_all = source.coords[rows]

    centre = source.coords[holdout_idx].mean(axis=0)
    region_r = np.sqrt(
        np.max(np.sum((source.coords[holdout_idx] - centre) ** 2, axis=1))
    )
    d = np.sqrt(np.sum((coords_all - centre) ** 2, axis=1))

    within = d <= context_radius_mult * region_r
    cand = rows[within]
    # ensure all held-out cells are present, then cap to patch_size by distance.
    cand = np.union1d(cand, holdout_idx)
    if cand.size > cfg.patch_size:
        dc = np.sqrt(np.sum((source.coords[cand] - centre) ** 2, axis=1))
        keep = np.argsort(dc, kind="stable")[: cfg.patch_size]
        cand = np.union1d(cand[keep], holdout_idx)  # never drop a held-out cell
        if cand.size > cfg.patch_size:              # held-out alone exceeds cap
            cand = holdout_idx.copy()

    coords = source.coords[cand]
    coords_n = normalise_coords(coords, cfg.coord_norm, cfg.knn)
    hold_set = set(int(i) for i in holdout_idx)
    mask = np.array([int(i) in hold_set for i in cand], dtype=bool)
    codes = {b.name: source.codes[b.name][cand] for b in source.branches}
    return Patch(
        global_idx=cand,
        coords_raw=coords,
        coords_norm=coords_n,
        codes=codes,
        mask=mask,
        section=s,
    )
