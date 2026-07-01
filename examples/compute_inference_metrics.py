"""
Post-inference metrics for SQUINT predicted AnnData.

Four metric families, each saved as a CSV in `<predict_dir>/metrics/`:

  1. Niche / cell-type identification (NMI, ARI)
       Discrete codebook indices vs. ground-truth obs labels. Each set
       of codes (cell / niche / legacy) is scored against EVERY label
       in `cell_label_keys ∪ niche_label_keys` — so the output covers
       both the diagonal (cell codes vs cell-type, niche codes vs niche
       labels — the codes' "designed" target) AND the cross
       (cell codes vs niche labels, niche codes vs cell-type — quantifies
       leak between the two branches; a healthy dual-VQ has high
       diagonal, low cross).
       Default label sets:
         cell  : {cell_type, cell_types}
         niche : {niche, Sub_molecular_tissue_region, ccf_region_name}
       Cells where the label is NaN are dropped before scoring; pairs
       with zero usable cells are skipped silently.

       Aggregate "niche" row: an extra `label_key="niche"` row is
       emitted per (split, code_key) carrying the cell-count-weighted
       average NMI/ARI across all niche_label_keys. This gives one
       comparable niche metric even when different AnnData sections
       in the inference output carry different niche label columns
       (e.g. some have `ccf_region_name`, others have
       `Sub_molecular_tissue_region`).

     When `adata.obs['data_split']` is present (stamped by the predict
     pipeline from the saved training config's `test_batches`), rows
     are also emitted stratified by split — `split = "all"`, `"train"`,
     and `"test"` — so train-set vs. held-out-test agreement can be
     read off the same CSV. Val cells are folded into "train" per the
     project convention (early-stopping val is in-distribution so it's
     part of the trained-on signal at inference time).

     RVQ / multi-level codebooks: the per-cell cluster id is the COMPOSITE
     index over all levels (`pd.factorize(tuple(per_level))`), which is
     the natural cluster assignment for residual quantizers (each unique
     tuple is a leaf cluster).

  2. Pearson reconstruction (cell + niche branch, gene-wise + cell-wise,
       log1p + raw, all genes + top-N HVG). Stratified by `data_split`
       when present. Mirrors the training-time `compute_benchmarking_metrics`
       set so train-time and post-inference numbers are directly
       comparable.

  3. Batch integration on continuous + quantized embeddings
       - iLISI  (kNN inverse Simpson; needs pynndescent; falls back to
                 a simple inline implementation if scib_metrics is missing)
       - ASW    (silhouette score on a sub-sample, sklearn)
       - MMD    (RBF, comparable bandwidth via median heuristic on
                 standardised embeddings; uses scipy.spatial.distance.cdist
                 to avoid the O(n^2 * d) memory blow-up of pairwise broadcast)

  4. Average cosine similarity per embedding (concentration sanity check)

Outputs:
    <predict_dir>/metrics/niche_identification_metrics.csv
    <predict_dir>/metrics/pearson_reconstruction_metrics.csv
    <predict_dir>/metrics/batch_integration_metrics.csv
    <predict_dir>/metrics/avg_cosine_similarity.csv
    <predict_dir>/metrics/analysis_summary.json    (consolidated)

Usage:
    python examples/compute_inference_metrics.py \\
        --predicted-adata <ARTIFACTS_DIR>/inference/<run>/predicted_adata.h5ad

    # Optional: override label / embedding lists
    python examples/compute_inference_metrics.py \\
        --predicted-adata <...> \\
        --cell-label-keys cell_type \\
        --niche-label-keys niche,Sub_molecular_tissue_region,ccf_region_name \\
        --emb-keys cell_emb,neighborhood_emb
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

# Silence two upstream FutureWarnings: dask's legacy DataFrame and
# anndata's deprecated `read_text` re-export. Both fire at import time
# of transitive dependencies; filter before they're imported.
warnings.filterwarnings(
    "ignore",
    message=r".*Importing read_text from `anndata` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.spatial.distance import cdist


def _to_dense_2d(x) -> np.ndarray:
    """
    Densify a matrix-like to a 2D numpy array.

    `np.asarray(scipy_sparse)` returns a 0-dim object array wrapping the
    sparse matrix instead of densifying — which crashes downstream
    fancy-indexing with cryptic "too many indices for 0-dimensional
    array" errors. Treat sparse explicitly via `.toarray()` and pass
    everything else through `np.asarray`.
    """
    if issparse(x):
        return x.toarray()
    return np.asarray(x)
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# Optional: pynndescent for fast approximate kNN used by iLISI.
try:
    from pynndescent import NNDescent
    _HAS_PYNNDESCENT = True
except Exception:
    _HAS_PYNNDESCENT = False

# Optional: scib_metrics' iLISI implementation. We fall back to a simple
# inline version if the package is unavailable so the script still runs
# in minimal environments.
try:
    from scib_metrics import ilisi_knn as _scib_ilisi_knn
    from scib_metrics.nearest_neighbors import NeighborsResults
    _HAS_SCIB_METRICS = True
except Exception:
    _HAS_SCIB_METRICS = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EMB_KEYS = [
    "cell_emb",
    "neighborhood_emb",
    "cell_latent",
    "neighborhood_latent",
    "X_squint",
    "X_squint_quantized",
]

# Cell-type label columns to score cell codes against. Tried in order;
# missing columns are silently skipped per-(seed, label) so the defaults
# can safely list keys from MULTIPLE datasets:
#   - "cell_type" / "cell_types": mmb (mouse brain), chl59 (CosMx Lung), squint_hln
#   - "annotation":               spatch_{ov, hcc, coad}_1p tissue subsets
#   - "new_annotation":           xhs1000 (Xenium human skin)
DEFAULT_CELL_LABEL_KEYS  = ["cell_type", "cell_types", "annotation", "new_annotation"]
# Niche / spatial-region label columns. Same per-dataset rationale:
#   - "niche":                                   chl59 (CosMx Lung), squint_hln
#   - "Sub_molecular_tissue_region", "ccf_region_name": mmb mouse brain
#   - "spatial_cluster":                         spatch_{ov, hcc, coad}_1p
#   - "niche_type":                              xhs1000 (Xenium human skin)
DEFAULT_NICHE_LABEL_KEYS = ["niche", "Sub_molecular_tissue_region", "ccf_region_name", "spatial_cluster", "niche_type"]


# ---------------------------------------------------------------------------
# Code-index extraction
# ---------------------------------------------------------------------------

def _flatten_codes(
        adata: ad.AnnData,
        obs_key: str,
    ) -> List[Tuple[np.ndarray, str]]:
    """
    Pull integer cluster id(s) per cell from `adata`.

    Single-level codebooks live in obs (1D); multi-level (RVQ /
    ConditionalVQ / multi-head) live in obsm (2D). For multi-level
    codebooks we report TWO views:

      - level-0 codes (`idx_2d[:, 0]`): the coarse macro cluster, the
        first level of the residual hierarchy. Useful for comparing
        macro-niche / macro-celltype agreement at low resolution.
      - composite cluster id (`pd.factorize(tuple(per-level))`): each
        unique tuple `(l1, l2, …, lL)` is a leaf cluster — the natural
        full-resolution assignment for an RVQ. Compares fine-grained
        agreement.

    Returns a list of (codes_1d, descriptive_name). Empty list if
    neither obs[obs_key] nor obsm[obs_key.replace('_index', '_indices')]
    exists. Single-level codebooks return exactly one entry; multi-
    level codebooks return two (level_0 + composite).
    """
    if obs_key in adata.obs.columns:
        return [(adata.obs[obs_key].to_numpy().astype(int), obs_key)]

    obsm_key = obs_key.replace("_index", "_indices")
    if obsm_key in adata.obsm:
        idx_2d = np.asarray(adata.obsm[obsm_key]).astype(int)
        if idx_2d.ndim == 1:
            return [(idx_2d, obsm_key)]
        out: List[Tuple[np.ndarray, str]] = []
        # level-0 (coarse macro cluster)
        out.append((idx_2d[:, 0].astype(int), f"{obsm_key}[level_0]"))
        # composite over all levels (fine-grained leaf cluster)
        composite, _uniq = pd.factorize(list(map(tuple, idx_2d)))
        out.append((composite.astype(int), f"{obsm_key}[composite]"))
        return out

    return []


# ---------------------------------------------------------------------------
# NMI / ARI
# ---------------------------------------------------------------------------

def compute_nmi_ari(
        adata: ad.AnnData,
        code_label_pairs: List[Tuple[np.ndarray, str, str]],
        cell_mask: Optional[np.ndarray] = None,
        split_label: str = "all",
    ) -> pd.DataFrame:
    """
    code_label_pairs: list of (codes_per_cell, code_name, label_obs_key).

    cell_mask, split_label:
        Optional 1D boolean array of length `adata.n_obs`. When provided,
        NMI/ARI are computed ONLY on cells where `cell_mask` is True. The
        emitted rows carry `split = split_label` ("all" / "train" /
        "test") for downstream filtering. Defaults compute on all cells.

    Returns a DataFrame with columns:
        split, code_key, label_key, NMI, ARI, n_cells,
        n_true_clusters, n_pred_clusters
    """
    if cell_mask is None:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    cell_mask = np.asarray(cell_mask, dtype=bool)
    if cell_mask.shape[0] != adata.n_obs:
        raise ValueError(
            f"cell_mask has length {cell_mask.shape[0]} but adata has "
            f"{adata.n_obs} cells."
        )
    if cell_mask.sum() == 0:
        print(f"  [{split_label}] no cells in this split; skipping all pairs")
        return pd.DataFrame()

    rows = []
    for codes, code_name, label_key in code_label_pairs:
        if label_key not in adata.obs.columns:
            print(f"  [{split_label}] skip ({code_name}, {label_key}): "
                  f"label column missing")
            continue
        labels = adata.obs[label_key]
        # Drop NaN AND the literal string "nan" (pandas converts None -> "None"
        # in object columns, and h5ad serialisation occasionally smuggles
        # "nan" strings through). Cast to str on a copy to detect both.
        labels_str = labels.astype("object").map(
            lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        )
        valid = (
            labels_str.notna() & (labels_str != "nan") & (labels_str != "None")
        ).to_numpy()
        # Combine label-validity with the split mask.
        keep = valid & cell_mask
        n = int(keep.sum())
        if n == 0:
            print(f"  [{split_label}] skip ({code_name}, {label_key}): "
                  f"no valid cells in this split")
            continue
        true = labels_str.to_numpy()[keep]
        pred = codes[keep]
        nmi = float(normalized_mutual_info_score(true, pred))
        ari = float(adjusted_rand_score(true, pred))
        n_true = int(len(np.unique(true)))
        n_pred = int(len(np.unique(pred)))
        print(
            f"  [{split_label}] {code_name:<28s} vs {label_key:<28s} "
            f"NMI={nmi:.4f}  ARI={ari:.4f}  "
            f"(n={n}, true_k={n_true}, pred_k={n_pred})"
        )
        rows.append({
            "split":            split_label,
            "code_key":         code_name,
            "label_key":        label_key,
            "NMI":              nmi,
            "ARI":              ari,
            "n_cells":          n,
            "n_true_clusters":  n_true,
            "n_pred_clusters":  n_pred,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pearson reconstruction metrics
# ---------------------------------------------------------------------------

def _pearson_pairwise(
        a: np.ndarray,
        b: np.ndarray,
        axis: int,
    ) -> np.ndarray:
    """
    Pairwise Pearson correlation between rows (axis=1) or columns (axis=0)
    of `a` and `b`. Returns a 1D array of correlations. NaN-safe: pairs
    with zero variance produce NaN (filtered out by the caller).

    Computed in float32 (was float64). Pearson correlation on count
    matrices doesn't need float64 precision — the relative error from
    float32 is ~1e-6, well below the noise floor of any biological
    measurement. The float64 upcast on a 1M x 946 matrix doubled
    memory pressure for no measurable benefit.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_mean = a.mean(axis=axis, keepdims=True)
    b_mean = b.mean(axis=axis, keepdims=True)
    a_c = a - a_mean
    b_c = b - b_mean
    num = (a_c * b_c).sum(axis=axis)
    den = np.sqrt((a_c ** 2).sum(axis=axis) * (b_c ** 2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def compute_pearson_metrics(
        adata: ad.AnnData,
        cell_mask: Optional[np.ndarray] = None,
        split_label: str = "all",
        log1p: bool = True,
        n_hvg: int = 50,
    ) -> pd.DataFrame:
    """
    Pearson correlation between predicted and observed expression on
    `cell_mask` cells. Two branches when both are present in the
    predicted adata:
      - cell branch:  X_hat   vs. X (raw counts)
      - niche branch: X_hat_nbr vs. X_nbr (1-hop neighborhood mean)

    Reported variants per branch (mirrors the training-time
    `compute_benchmarking_metrics`):
      - axis:        gene_wise (Pearson per gene over cells, then
                                aggregated)
                  +  cell_wise (Pearson per cell over genes, then
                                aggregated)
      - transform:   log1p     (default; robust to mean-count dominance)
                  +  raw       (raw counts; tracked for back-compat)
      - gene_subset: all       (every gene in `var`)
                  +  hvg{N}    (top-N highly-variable genes by per-gene
                                variance on the log1p target — N defaults
                                to 50 to match the training metrics)
    `cell_wise × hvg{N}` is intentionally NOT reported (Pearson per cell
    over only N genes is statistically noisy and the training-time
    metric set agrees).

    Returns one row per (branch, axis, transform, gene_subset) for the
    given `split_label`; columns:
        split, branch, axis, transform, gene_subset,
        pearson_mean, pearson_median, n_cells, n_genes
    """
    if cell_mask is None:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    cell_mask = np.asarray(cell_mask, dtype=bool)
    if cell_mask.sum() == 0:
        return pd.DataFrame()

    rows: List[dict] = []
    # Cache the dense (target, pred) pairs across calls keyed by
    # `id(adata)`. compute_pearson_metrics is invoked once per
    # `data_split` value (all / train / test) — without this cache each
    # call re-densifies `adata.X` (potentially sparse, multi-GB on
    # mmb20) plus the three predicted layers, even though the dense
    # contents are identical. The split-specific subset happens AFTER
    # densification.
    cache_key = id(adata)
    if not hasattr(compute_pearson_metrics, "_dense_cache"):
        compute_pearson_metrics._dense_cache = {}
    cache = compute_pearson_metrics._dense_cache.get(cache_key)
    if cache is None:
        cache = {}
        compute_pearson_metrics._dense_cache[cache_key] = cache
    for branch, pred_key in [
        ("cell",  "X_hat"),
        ("niche", "X_hat_nbr"),
    ]:
        # Cell-branch ground truth lives in adata.X; niche-branch lives in
        # adata.layers["X_nbr"] (or adata.uns["X_nbr"] for back-compat).
        # Both predicted tensors live in adata.layers.
        if (branch, "target") not in cache:
            if branch == "cell":
                cache[(branch, "target")] = _to_dense_2d(adata.X)
            elif "X_nbr" in adata.layers:
                cache[(branch, "target")] = _to_dense_2d(adata.layers["X_nbr"])
            elif "X_nbr" in adata.uns:
                cache[(branch, "target")] = _to_dense_2d(adata.uns["X_nbr"])
            else:
                cache[(branch, "target")] = None
            if pred_key in adata.layers:
                cache[(branch, "pred")] = _to_dense_2d(adata.layers[pred_key])
            else:
                cache[(branch, "pred")] = None
        if cache[(branch, "target")] is None or cache[(branch, "pred")] is None:
            continue
        target_full = cache[(branch, "target")]
        pred_full   = cache[(branch, "pred")]

        # Subset to cells in this split (view, not copy, where possible).
        target = target_full[cell_mask]
        pred   = pred_full[cell_mask]
        if target.size == 0:
            continue
        n_cells, n_genes = target.shape

        # HVG indices: top-N genes by variance on the log1p target. Only
        # used for the `gene_wise × hvg{N}` rows; computed once per branch.
        n_hvg_eff = min(int(n_hvg), n_genes)
        if n_hvg_eff > 0:
            target_log_for_hvg = np.log1p(np.clip(target, 0, None))
            gene_var = target_log_for_hvg.var(axis=0)
            hvg_idx = np.argpartition(-gene_var, n_hvg_eff - 1)[:n_hvg_eff]
            hvg_idx.sort()
        else:
            hvg_idx = np.array([], dtype=int)

        # Iterate (transform, axis, gene_subset). Skip the cell_wise ×
        # hvg subset combination intentionally.
        transforms = (("log1p",) if log1p else ()) + ("raw",)
        for transform in transforms:
            if transform == "log1p":
                t_full = np.log1p(np.clip(target, 0, None))
                p_full = np.log1p(np.clip(pred,   0, None))
            else:
                t_full, p_full = target, pred

            for axis_name, axis in (("gene_wise", 0), ("cell_wise", 1)):
                gene_subsets = ["all"]
                if axis_name == "gene_wise" and n_hvg_eff > 0:
                    gene_subsets.append(f"hvg{n_hvg_eff}")

                for gene_subset in gene_subsets:
                    if gene_subset == "all":
                        t_sub, p_sub = t_full, p_full
                        n_genes_sub = n_genes
                    else:
                        t_sub = t_full[:, hvg_idx]
                        p_sub = p_full[:, hvg_idx]
                        n_genes_sub = int(hvg_idx.size)

                    vec = _pearson_pairwise(p_sub, t_sub, axis=axis)
                    vec = vec[np.isfinite(vec)]
                    if vec.size == 0:
                        continue
                    rows.append({
                        "split":            split_label,
                        "branch":           branch,
                        "axis":             axis_name,
                        "transform":        transform,
                        "gene_subset":      gene_subset,
                        "pearson_mean":     float(vec.mean()),
                        "pearson_median":   float(np.median(vec)),
                        "n_cells":          int(n_cells),
                        "n_genes":          n_genes_sub,
                    })
                    print(
                        f"  [{split_label}] {branch:<5s} {axis_name:<9s} "
                        f"{transform:<5s} {gene_subset:<7s} "
                        f"mean={float(vec.mean()):.4f}  "
                        f"median={float(np.median(vec)):.4f}  "
                        f"(n_cells={n_cells}, n_genes={n_genes_sub})"
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch integration
# ---------------------------------------------------------------------------

def _ilisi_inline(
        knn_indices: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
    """
    Inverse Simpson Index per cell over its kNN's batch labels (uniform
    weighting, no Gaussian re-weighting). Higher = more batch-mixed.
    Equivalent to scib's iLISI up to the kernel weighting used.

    Vectorised with `np.add.at` over (cell, batch) pairs (was a Python
    `for i in range(n)` loop calling `np.unique` per cell). For 1M
    cells × 30 neighbours that's ~30M Python iterations → minutes;
    vectorised version is ~50-100× faster.
    """
    nbr_batches = batch_labels[knn_indices]  # (n, k)
    n, k = nbr_batches.shape
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Densify batch labels to contiguous integer ids in [0, n_batches).
    unique_batches, dense = np.unique(nbr_batches, return_inverse=True)
    dense = dense.reshape(n, k)
    n_batches = int(unique_batches.size)

    # Build a (n, n_batches) count matrix in one vectorised scatter.
    counts = np.zeros((n, n_batches), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    cols = dense.ravel()
    np.add.at(counts, (rows, cols), 1.0)

    p = counts / float(k)             # uniform weighting → divide by k
    inv_simpson = 1.0 / np.sum(p ** 2, axis=1)
    return inv_simpson


def compute_ilisi(
        emb: np.ndarray,
        batch: np.ndarray,
        n_neighbors: int = 90,
    ) -> Optional[float]:
    """
    Mean iLISI across all cells. Uses pynndescent for the kNN graph
    (much faster than sklearn for large n). Falls back to a simple
    inline iLISI when scib_metrics isn't installed.
    """
    if not _HAS_PYNNDESCENT:
        print("    iLISI: pynndescent not installed; skipping")
        return None
    knn = NNDescent(emb, n_neighbors=min(n_neighbors, len(emb) - 1))
    indices, distances = knn.neighbor_graph

    if _HAS_SCIB_METRICS:
        nb = NeighborsResults(indices=indices, distances=distances)
        return float(np.mean(_scib_ilisi_knn(nb, batch)))

    # Fallback: uniform-weighted iLISI on the kNN indices.
    return float(np.mean(_ilisi_inline(indices, batch)))


def compute_mmd_comparable(
        emb: np.ndarray,
        batch: np.ndarray,
        n_sub: int = 2000,
        n_sigma: int = 1000,
        rng: Optional[np.random.Generator] = None,
        max_pairs: Optional[int] = None,
    ) -> Optional[float]:
    """
    MMD with median-heuristic RBF bandwidth on standardised embeddings.
    Uses scipy.spatial.distance.cdist block-wise (avoids the O(n^2 * d) memory
    blow-up of broadcasting).

    Multi-batch: for 2 batches this is the two-sample MMD; for >2 batches it is
    the MEAN of the pairwise MMD over all unordered batch pairs (the standard
    multi-batch summary — reduces EXACTLY to the 2-batch value). The RBF
    bandwidth is fixed once (global median heuristic) so every pair is on the
    same scale. Returns None with <2 non-empty batches. `max_pairs` (optional)
    subsamples the pair set to bound cost when there are very many batches.
    """
    rng = rng or np.random.default_rng(0)
    X = StandardScaler().fit_transform(emb)

    # sigma via median heuristic on a subsample (median of pairwise distances).
    # Global (shared across all pairs) so the MMD scale is comparable.
    idx = rng.choice(len(X), min(n_sigma, len(X)), replace=False)
    sub = X[idx]
    dists = cdist(sub, sub)
    nz = dists[dists > 0]
    sigma = float(np.median(nz)) if nz.size else 1.0
    inv2sigma2 = 1.0 / (2.0 * sigma * sigma)

    def rbf_mean(A: np.ndarray, B: np.ndarray) -> float:
        d2 = cdist(A, B, metric="sqeuclidean")
        return float(np.exp(-d2 * inv2sigma2).mean())

    # one sub-sample per NON-EMPTY batch (draw order == np.unique order, so the
    # 2-batch result is bit-identical to the previous implementation)
    subs: List[np.ndarray] = []
    for c in np.unique(batch):
        bc = X[batch == c]
        if len(bc) == 0:
            continue
        j = rng.choice(len(bc), min(n_sub, len(bc)), replace=False)
        subs.append(bc[j])
    if len(subs) < 2:
        print(f"    MMD: needs >=2 non-empty batches, got {len(subs)}; skipping")
        return None

    # precompute the within-batch kernel means once (reused across pairs)
    self_k = [rbf_mean(s, s) for s in subs]
    pairs = [(i, j) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    if max_pairs is not None and 0 < max_pairs < len(pairs):
        keep = sorted(rng.choice(len(pairs), max_pairs, replace=False).tolist())
        pairs = [pairs[k] for k in keep]
    vals = [self_k[i] + self_k[j] - 2.0 * rbf_mean(subs[i], subs[j])
            for (i, j) in pairs]
    if len(subs) > 2:
        print(f"    MMD: mean over {len(pairs)} batch pairs ({len(subs)} batches)")
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Average cosine similarity
# ---------------------------------------------------------------------------

def compute_avg_cosine(emb: np.ndarray) -> float:
    """
    Average pairwise cosine similarity over all unordered cell pairs.
    Closed form via row-normalised sum: avg_cos = (||S||^2 - n) / (n * (n-1))
    where S = sum of L2-normalised rows. O(n) memory, O(n*d) time.

    Computed in float32 (was float64). The closed-form sum of unit
    vectors is numerically well-conditioned at this scale (~1M rows ×
    ~256 dim); the relative error from float32 is ~1e-5, well below
    the noise floor of any biological metric. Avoids a 1M × D float64
    copy per emb_key (cumulatively several GB of churn across the 6
    embedding slots).
    """
    X = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-12)
    n = X.shape[0]
    if n < 2:
        return float("nan")
    S = X.sum(axis=0)
    return float((S @ S - n) / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--predicted-adata", type=str, required=True,
        help="Path to predicted_adata.h5ad written by --predict.",
    )
    ap.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: <predicted-adata-dir>/metrics).",
    )
    ap.add_argument(
        "--batch-key", type=str, default="adata_batch_id",
        help="Obs column used as batch label for batch integration "
             "metrics (default: adata_batch_id).",
    )
    ap.add_argument(
        "--emb-keys", type=str, default=",".join(DEFAULT_EMB_KEYS),
        help="Comma-separated obsm keys for batch integration / cosine "
             "similarity. Missing keys are skipped silently.",
    )
    ap.add_argument(
        "--cell-code-key", type=str, default="cell_code_index",
        help="Obs key (1D) for cell-branch codes; falls back to obsm with "
             "key '_index' replaced by '_indices' for multi-level codes "
             "(composite cluster id via factorize). Set to '' to skip "
             "cell-branch NMI/ARI.",
    )
    ap.add_argument(
        "--niche-code-key", type=str, default="neighborhood_code_index",
        help="Obs key (1D) for niche-branch codes. Same fallback as above.",
    )
    ap.add_argument(
        "--legacy-code-key", type=str, default="code_index",
        help="Single-codebook fallback (legacy VQNiche). Same fallback as above.",
    )
    ap.add_argument(
        "--cell-label-keys", type=str,
        default=",".join(DEFAULT_CELL_LABEL_KEYS),
        help="Comma-separated obs columns to compare cell codes against.",
    )
    ap.add_argument(
        "--niche-label-keys", type=str,
        default=",".join(DEFAULT_NICHE_LABEL_KEYS),
        help="Comma-separated obs columns to compare niche codes against.",
    )
    ap.add_argument(
        "--ilisi-n-neighbors", type=int, default=90,
        help="kNN size for iLISI (default 90).",
    )
    ap.add_argument(
        "--ilisi-max-cells", type=int, default=100000,
        help="Cap the number of cells used for iLISI; if the embedding has "
             "more, a random subsample of this size is used (an unbiased "
             "estimate of mean iLISI). 0 = no cap. iLISI builds a FULL kNN "
             "graph with no internal subsampling (unlike MMD/ASW), so on large "
             "sections (e.g. ~700k cells) it OOMs and kills the whole metrics "
             "step — capping it keeps memory bounded (default 100000).",
    )
    ap.add_argument(
        "--asw-sample-size", type=int, default=10000,
        help="Sub-sample size for sklearn silhouette_score (default 10000).",
    )
    ap.add_argument(
        "--mmd-n-sub", type=int, default=2000,
        help="Per-batch sub-sample size for MMD (default 2000).",
    )
    ap.add_argument(
        "--mmd-n-sigma", type=int, default=1000,
        help="Sub-sample for MMD bandwidth median heuristic (default 1000).",
    )
    ap.add_argument(
        "--mmd-max-pairs", type=int, default=0,
        help="For >2 batches, MMD = mean over batch pairs. Cap the number of "
             "pairs sampled (0 = all pairs). Use for datasets with many "
             "batches (e.g. smb1-20b: 20 batches = 190 pairs).",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for sub-samples in batch integration metrics.",
    )
    args = ap.parse_args()

    adata_path = Path(args.predicted_adata)
    adata = ad.read_h5ad(adata_path)
    print(f"Loaded {adata_path}")
    print(f"  n_obs={adata.n_obs}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    out_dir = Path(args.out_dir) if args.out_dir else adata_path.parent / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_label_keys  = [k.strip() for k in args.cell_label_keys.split(",")  if k.strip()]
    niche_label_keys = [k.strip() for k in args.niche_label_keys.split(",") if k.strip()]
    emb_keys         = [k.strip() for k in args.emb_keys.split(",")         if k.strip()]

    # ---------------------------------------------------------------- NMI/ARI
    print("\n=== Niche / cell-type identification (NMI, ARI) ===")
    code_label_pairs: List[Tuple[np.ndarray, str, str]] = []

    # Score each branch's codes against BOTH label families (cell-type
    # AND niche). The "natural" pairing — cell codes vs cell-type labels,
    # niche codes vs niche labels — is the diagonal: codes should match
    # the labels they were designed to capture. The cross pairing — cell
    # codes vs niche labels, niche codes vs cell-type labels — is also
    # informative: it quantifies how much "leak" there is between
    # branches (a healthy dual-VQ should have HIGH diagonal, LOW cross).
    # Combine both label families into one de-duplicated list so each
    # set of codes gets scored against every available label key.
    seen: set = set()
    all_label_keys: List[str] = []
    for lab in cell_label_keys + niche_label_keys:
        if lab and lab not in seen:
            seen.add(lab)
            all_label_keys.append(lab)

    if args.cell_code_key:
        cell_views = _flatten_codes(adata, args.cell_code_key)
        if cell_views:
            for codes, name in cell_views:
                for lab in all_label_keys:
                    code_label_pairs.append((codes, name, lab))
        else:
            print(f"  cell-branch codes not found "
                  f"(tried obs[{args.cell_code_key!r}] and the obsm fallback)")

    if args.niche_code_key:
        niche_views = _flatten_codes(adata, args.niche_code_key)
        if niche_views:
            for codes, name in niche_views:
                for lab in all_label_keys:
                    code_label_pairs.append((codes, name, lab))
        else:
            print(f"  niche-branch codes not found "
                  f"(tried obs[{args.niche_code_key!r}] and the obsm fallback)")

    if args.legacy_code_key:
        legacy_views = _flatten_codes(adata, args.legacy_code_key)
        if legacy_views:
            for codes, name in legacy_views:
                for lab in all_label_keys:
                    code_label_pairs.append((codes, name, lab))

    # Always compute on the full adata ("all"). Additionally, when the
    # predict pipeline stamped `data_split` (train/test split inferred
    # from the saved training config's `test_batches`), emit per-split
    # rows so the user can read off train- vs test-set NMI/ARI directly.
    # Per the user spec, val cells are folded into "train" upstream — so
    # only "train" and "test" splits exist (in addition to "all").
    nmi_ari_frames: List[pd.DataFrame] = [
        compute_nmi_ari(adata, code_label_pairs, cell_mask=None, split_label="all"),
    ]
    if "data_split" in adata.obs.columns:
        split_col = adata.obs["data_split"].astype("object")
        for split_name in ("train", "test"):
            mask = (split_col == split_name).to_numpy()
            if mask.any():
                nmi_ari_frames.append(
                    compute_nmi_ari(
                        adata, code_label_pairs,
                        cell_mask=mask, split_label=split_name,
                    )
                )
    nmi_ari_df = pd.concat(
        [df for df in nmi_ari_frames if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in nmi_ari_frames) else pd.DataFrame()

    # Aggregate niche labels into a single weighted "niche" row.
    # Rationale: different AnnData objects in the inference output may
    # carry DIFFERENT niche label columns (e.g. one section uses
    # `ccf_region_name`, another uses `Sub_molecular_tissue_region`). A
    # cell-count-weighted average across all niche_label_keys yields one
    # comparable "niche" NMI/ARI per (split, code_key) regardless of
    # which specific column was populated for which cells. The
    # n_cells weighting matters because each per-label row already
    # restricts to the cells where THAT label is non-NaN, so n varies.
    if not nmi_ari_df.empty and niche_label_keys:
        agg_rows: List[dict] = []
        niche_set = set(niche_label_keys)
        for (split, code_key), grp in nmi_ari_df[
            nmi_ari_df["label_key"].isin(niche_set)
        ].groupby(["split", "code_key"], sort=False):
            n_cells_total = int(grp["n_cells"].sum())
            if n_cells_total <= 0:
                continue
            w_nmi = float((grp["NMI"] * grp["n_cells"]).sum() / n_cells_total)
            w_ari = float((grp["ARI"] * grp["n_cells"]).sum() / n_cells_total)
            print(
                f"  [{split}] {code_key:<28s} vs {'niche (weighted)':<28s} "
                f"NMI={w_nmi:.4f}  ARI={w_ari:.4f}  "
                f"(n={n_cells_total}, "
                f"sources={list(grp['label_key'])})"
            )
            agg_rows.append({
                "split":            split,
                "code_key":         code_key,
                "label_key":        "niche",
                "NMI":              w_nmi,
                "ARI":              w_ari,
                "n_cells":          n_cells_total,
                # Aggregate doesn't have a meaningful single
                # n_true_clusters / n_pred_clusters — leave NaN to make
                # it obvious this row is an aggregate.
                "n_true_clusters":  pd.NA,
                "n_pred_clusters":  pd.NA,
            })
        if agg_rows:
            nmi_ari_df = pd.concat(
                [nmi_ari_df, pd.DataFrame(agg_rows)],
                ignore_index=True,
            )

    if not nmi_ari_df.empty:
        nmi_ari_csv = out_dir / "niche_identification_metrics.csv"
        nmi_ari_df.to_csv(nmi_ari_csv, index=False)
        print(f"  -> {nmi_ari_csv}")
    else:
        print("  (no valid (code, label) pairs to score)")

    # ----------------------------------------------- Pearson reconstruction
    print("\n=== Pearson reconstruction (cell + niche branch) ===")
    pearson_frames: List[pd.DataFrame] = [
        compute_pearson_metrics(adata, cell_mask=None, split_label="all"),
    ]
    if "data_split" in adata.obs.columns:
        split_col = adata.obs["data_split"].astype("object")
        for split_name in ("train", "test"):
            mask = (split_col == split_name).to_numpy()
            if mask.any():
                pearson_frames.append(
                    compute_pearson_metrics(
                        adata, cell_mask=mask, split_label=split_name,
                    )
                )
    pearson_df = pd.concat(
        [df for df in pearson_frames if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in pearson_frames) else pd.DataFrame()
    if not pearson_df.empty:
        pearson_csv = out_dir / "pearson_reconstruction_metrics.csv"
        pearson_df.to_csv(pearson_csv, index=False)
        print(f"  -> {pearson_csv}")
    else:
        print("  (no Pearson metrics computed — X_hat / X_hat_nbr / X_nbr "
              "may be missing from the predicted adata)")

    # -------------------------------------------------------- Batch integration
    print("\n=== Batch integration (iLISI, ASW, MMD) ===")
    if args.batch_key not in adata.obs.columns:
        print(f"  batch key {args.batch_key!r} missing from obs; "
              f"skipping all batch integration metrics")
        batch_int_df = pd.DataFrame()
    else:
        batch = adata.obs[args.batch_key].to_numpy()
        # iLISI / silhouette / MMD all want a 1D label vector — coerce strings.
        if batch.dtype.kind not in ("i", "u"):
            batch = np.asarray([str(b) for b in batch])
        rows = []
        for emb_key in emb_keys:
            if emb_key not in adata.obsm:
                continue
            emb = np.asarray(adata.obsm[emb_key])
            if emb.dtype not in (np.float32, np.float64):
                emb = emb.astype(np.float32)
            if emb.ndim != 2 or emb.shape[1] == 0:
                print(f"  [{emb_key}] not a 2D embedding (shape={emb.shape}); skipping")
                continue
            print(f"  [{emb_key}]")

            # iLISI builds a FULL kNN graph (no internal subsampling) — it OOMs
            # on large sections and, being un-guarded, would kill the whole
            # metrics step before the CSV is written (resolution metrics, run
            # earlier, survive — exactly the "no integration metrics" symptom).
            # Cap the cell count (random subsample = unbiased mean-iLISI
            # estimate) and guard so a failure can't abort MMD/ASW + the write.
            try:
                if args.ilisi_max_cells and len(emb) > args.ilisi_max_cells:
                    _sub = np.random.default_rng(args.seed).choice(
                        len(emb), args.ilisi_max_cells, replace=False)
                    print(f"    iLISI: subsampling {len(emb)} -> "
                          f"{args.ilisi_max_cells} cells")
                    ilisi = compute_ilisi(emb[_sub], batch[_sub],
                                          n_neighbors=args.ilisi_n_neighbors)
                else:
                    ilisi = compute_ilisi(emb, batch,
                                          n_neighbors=args.ilisi_n_neighbors)
            except Exception as exc:
                print(f"    iLISI: failed ({exc}); skipping")
                ilisi = None
            try:
                asw = float(silhouette_score(
                    emb, batch, metric="euclidean",
                    sample_size=min(args.asw_sample_size, len(emb)),
                    random_state=args.seed,
                ))
            except Exception as exc:
                print(f"    ASW: failed ({exc}); skipping")
                asw = None
            mmd = compute_mmd_comparable(
                emb, batch,
                n_sub=args.mmd_n_sub, n_sigma=args.mmd_n_sigma,
                rng=np.random.default_rng(args.seed),
                max_pairs=(args.mmd_max_pairs or None),
            )

            if ilisi is not None:
                rows.append({"emb_key": emb_key, "metric": "iLISI", "score": ilisi})
                print(f"    iLISI = {ilisi:.4f}")
            if asw is not None:
                rows.append({"emb_key": emb_key, "metric": "ASW",   "score": asw})
                print(f"    ASW   = {asw:.4f}")
            if mmd is not None:
                rows.append({"emb_key": emb_key, "metric": "MMD",   "score": mmd})
                print(f"    MMD   = {mmd:.4f}")
        batch_int_df = pd.DataFrame(rows)
        if not batch_int_df.empty:
            csv = out_dir / "batch_integration_metrics.csv"
            batch_int_df.to_csv(csv, index=False)
            print(f"  -> {csv}")
        else:
            print("  (no embeddings produced batch integration scores)")

    # ----------------------------------------------- Average cosine similarity
    print("\n=== Average cosine similarity ===")
    rows = []
    for emb_key in emb_keys:
        if emb_key not in adata.obsm:
            continue
        emb = np.asarray(adata.obsm[emb_key])
        if emb.ndim != 2 or emb.shape[1] == 0:
            continue
        avg_cos = compute_avg_cosine(emb)
        print(f"  [{emb_key}] avg_cos = {avg_cos:.4f}")
        rows.append({"emb_key": emb_key, "avg_cosine_similarity": avg_cos})
    cos_df = pd.DataFrame(rows)
    if not cos_df.empty:
        csv = out_dir / "avg_cosine_similarity.csv"
        cos_df.to_csv(csv, index=False)
        print(f"  -> {csv}")
    else:
        print("  (no embeddings to score)")

    # ------------------------------------------------------- consolidated JSON
    summary = {
        "predicted_adata":                  str(adata_path),
        "n_obs":                            int(adata.n_obs),
        "batch_key":                        args.batch_key,
        "niche_identification_metrics":     nmi_ari_df.to_dict(orient="records"),
        "pearson_reconstruction_metrics":   pearson_df.to_dict(orient="records"),
        "batch_integration_metrics":        batch_int_df.to_dict(orient="records"),
        "avg_cosine_similarity":            cos_df.to_dict(orient="records"),
    }
    summary_path = out_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"\nDone. All metrics in {out_dir}")


if __name__ == "__main__":
    main()
