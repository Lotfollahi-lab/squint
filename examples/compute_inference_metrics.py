"""
Post-inference metrics for SQUINT predicted AnnData.

Three metric families, each saved as a CSV in `<predict_dir>/metrics/`:

  1. Niche / cell-type identification (NMI, ARI)
       Discrete codebook indices vs. ground-truth obs labels.
       - cell codes      vs. {cell_type, cell_types}
       - niche codes     vs. {niche, Sub_molecular_tissue_region,
                              ccf_region_name}
       - legacy codes    vs. all of the above
     Cells where the label is NaN are dropped before scoring; pairs with
     zero usable cells are skipped silently.

     RVQ / multi-level codebooks: the per-cell cluster id is the COMPOSITE
     index over all levels (`pd.factorize(tuple(per_level))`), which is
     the natural cluster assignment for residual quantizers (each unique
     tuple is a leaf cluster).

  2. Batch integration on continuous + quantized embeddings
       - iLISI  (kNN inverse Simpson; needs pynndescent; falls back to
                 a simple inline implementation if scib_metrics is missing)
       - ASW    (silhouette score on a sub-sample, sklearn)
       - MMD    (RBF, comparable bandwidth via median heuristic on
                 standardised embeddings; uses scipy.spatial.distance.cdist
                 to avoid the O(n^2 * d) memory blow-up of pairwise broadcast)

  3. Average cosine similarity per embedding (concentration sanity check)

Outputs:
    <predict_dir>/metrics/niche_identification_metrics.csv
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
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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

DEFAULT_CELL_LABEL_KEYS  = ["cell_type", "cell_types"]
DEFAULT_NICHE_LABEL_KEYS = ["niche", "Sub_molecular_tissue_region", "ccf_region_name"]


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
    ) -> pd.DataFrame:
    """
    code_label_pairs: list of (codes_per_cell, code_name, label_obs_key).
    Returns a DataFrame with columns:
        code_key, label_key, NMI, ARI, n_cells, n_true_clusters, n_pred_clusters
    """
    rows = []
    for codes, code_name, label_key in code_label_pairs:
        if label_key not in adata.obs.columns:
            print(f"  skip ({code_name}, {label_key}): label column missing")
            continue
        labels = adata.obs[label_key]
        # Drop NaN AND the literal string "nan" (pandas converts None -> "None"
        # in object columns, and h5ad serialisation occasionally smuggles
        # "nan" strings through). Cast to str on a copy to detect both.
        labels_str = labels.astype("object").map(
            lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        )
        mask = labels_str.notna() & (labels_str != "nan") & (labels_str != "None")
        n = int(mask.sum())
        if n == 0:
            print(f"  skip ({code_name}, {label_key}): all cells NaN/missing")
            continue
        true = labels_str.loc[mask].to_numpy()
        pred = codes[mask.to_numpy()]
        nmi = float(normalized_mutual_info_score(true, pred))
        ari = float(adjusted_rand_score(true, pred))
        n_true = int(len(np.unique(true)))
        n_pred = int(len(np.unique(pred)))
        print(
            f"  {code_name:<28s} vs {label_key:<28s} "
            f"NMI={nmi:.4f}  ARI={ari:.4f}  "
            f"(n={n}, true_k={n_true}, pred_k={n_pred})"
        )
        rows.append({
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
    """
    nbr_batches = batch_labels[knn_indices]  # (n, k)
    n = nbr_batches.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        _, counts = np.unique(nbr_batches[i], return_counts=True)
        p = counts / counts.sum()
        out[i] = 1.0 / np.sum(p ** 2)
    return out


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
    ) -> Optional[float]:
    """
    MMD with median-heuristic RBF bandwidth on standardised embeddings.
    Uses scipy.spatial.distance.cdist to compute pairwise distances
    block-wise — avoids the O(n^2 * d) memory blow-up of broadcasting.
    Defined for exactly two batches; returns None otherwise.
    """
    rng = rng or np.random.default_rng(0)
    X = StandardScaler().fit_transform(emb)

    # sigma via median heuristic on a subsample (median of pairwise distances)
    idx = rng.choice(len(X), min(n_sigma, len(X)), replace=False)
    sub = X[idx]
    dists = cdist(sub, sub)
    nz = dists[dists > 0]
    sigma = float(np.median(nz)) if nz.size else 1.0

    cats = np.unique(batch)
    if len(cats) != 2:
        print(f"    MMD: needs exactly 2 batches, got {len(cats)}; skipping")
        return None
    mask0 = batch == cats[0]
    mask1 = batch == cats[1]
    b0 = X[mask0]
    b1 = X[mask1]
    if len(b0) == 0 or len(b1) == 0:
        print(f"    MMD: a batch is empty; skipping")
        return None

    idx0 = rng.choice(len(b0), min(n_sub, len(b0)), replace=False)
    idx1 = rng.choice(len(b1), min(n_sub, len(b1)), replace=False)
    b0s, b1s = b0[idx0], b1[idx1]

    inv2sigma2 = 1.0 / (2.0 * sigma * sigma)

    def rbf_mean(A: np.ndarray, B: np.ndarray) -> float:
        d2 = cdist(A, B, metric="sqeuclidean")
        return float(np.exp(-d2 * inv2sigma2).mean())

    return rbf_mean(b0s, b0s) + rbf_mean(b1s, b1s) - 2.0 * rbf_mean(b0s, b1s)


# ---------------------------------------------------------------------------
# Average cosine similarity
# ---------------------------------------------------------------------------

def compute_avg_cosine(emb: np.ndarray) -> float:
    """
    Average pairwise cosine similarity over all unordered cell pairs.
    Closed form via row-normalised sum: avg_cos = (||S||^2 - n) / (n * (n-1))
    where S = sum of L2-normalised rows. O(n) memory, O(n*d) time.
    """
    X = emb.astype(np.float64)
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

    if args.cell_code_key:
        cell_views = _flatten_codes(adata, args.cell_code_key)
        if cell_views:
            for codes, name in cell_views:
                for lab in cell_label_keys:
                    code_label_pairs.append((codes, name, lab))
        else:
            print(f"  cell-branch codes not found "
                  f"(tried obs[{args.cell_code_key!r}] and the obsm fallback)")

    if args.niche_code_key:
        niche_views = _flatten_codes(adata, args.niche_code_key)
        if niche_views:
            for codes, name in niche_views:
                for lab in niche_label_keys:
                    code_label_pairs.append((codes, name, lab))
        else:
            print(f"  niche-branch codes not found "
                  f"(tried obs[{args.niche_code_key!r}] and the obsm fallback)")

    if args.legacy_code_key:
        legacy_views = _flatten_codes(adata, args.legacy_code_key)
        if legacy_views:
            seen = set()
            all_labels = []
            for lab in cell_label_keys + niche_label_keys:
                if lab not in seen:
                    seen.add(lab)
                    all_labels.append(lab)
            for codes, name in legacy_views:
                for lab in all_labels:
                    code_label_pairs.append((codes, name, lab))

    nmi_ari_df = compute_nmi_ari(adata, code_label_pairs)
    if not nmi_ari_df.empty:
        nmi_ari_csv = out_dir / "niche_identification_metrics.csv"
        nmi_ari_df.to_csv(nmi_ari_csv, index=False)
        print(f"  -> {nmi_ari_csv}")
    else:
        print("  (no valid (code, label) pairs to score)")

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

            ilisi = compute_ilisi(emb, batch, n_neighbors=args.ilisi_n_neighbors)
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
        "predicted_adata":              str(adata_path),
        "n_obs":                        int(adata.n_obs),
        "batch_key":                    args.batch_key,
        "niche_identification_metrics": nmi_ari_df.to_dict(orient="records"),
        "batch_integration_metrics":    batch_int_df.to_dict(orient="records"),
        "avg_cosine_similarity":        cos_df.to_dict(orient="records"),
    }
    summary_path = out_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"\nDone. All metrics in {out_dir}")


if __name__ == "__main__":
    main()
