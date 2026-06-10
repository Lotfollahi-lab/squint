"""
UMAP plots of SQUINT latents from a predicted AnnData.

For each latent slot present in the predicted AnnData, builds a kNN
graph in latent space, runs UMAP, and writes one SVG per (latent,
color-key) combination to <inference dir>/umap_plots/.

Latents searched (in obsm):
    cell_emb              quantized cell-branch embedding (z_q_cell)
    neighborhood_emb      quantized niche-branch embedding (z_q_niche)
    cell_latent           continuous pre-VQ cell latent (z_mlp)
    neighborhood_latent   continuous pre-VQ niche latent (z_gnn)
    X_squint              legacy single-VQ continuous latent (z_latent)
    X_squint_quantized    legacy single-VQ quantized latent (z_q)

Color keys:
    --batch-key           required: 1 panel per latent showing batch
                          mixing (default 'adata_batch_id'). When the
                          dataset has exactly two batches, the legend
                          gets renamed via --batch-rename (e.g.
                          'MERFISH,STARmap PLUS' for the brain dataset).
    --label-keys          comma-separated obs columns to plot (e.g.
                          'cell_type,niche'). Each yields one extra
                          panel per latent.

Usage:
    python examples/plot_latent_umap.py \\
        --predicted-adata <ARTIFACTS_DIR>/inference/<run>/predicted_adata.h5ad \\
        --batch-key adata_batch_id \\
        --label-keys cell_type \\
        --batch-rename "MERFISH,STARmap PLUS"
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

# Silence two upstream FutureWarnings (dask legacy DataFrame, anndata
# `read_text` re-export); see run_squint.py for context.
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
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

# Optional GPU acceleration: rapids-singlecell ships drop-in replacements
# for sc.pp.neighbors / sc.tl.umap (cuML cuKNN + cuML UMAP). We auto-
# detect: if rapids-singlecell imports AND a GPU is visible, run on GPU;
# otherwise fall back to scanpy CPU. Override with --cpu-only / --use-gpu.
def _detect_gpu_umap_backend() -> str:
    try:
        import rapids_singlecell  # noqa: F401
        import cupy  # noqa: F401
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return "rapids"
    except Exception:  # noqa: BLE001 — any import / runtime failure -> CPU
        pass
    return "scanpy"


def _compute_umap(
        adata: ad.AnnData,
        emb_key: str,
        n_neighbors: int,
        n_pcs: Optional[int],
        key_added: str,
        backend: str,
    ) -> None:
    """Build kNN graph + UMAP coordinates IN-PLACE on `adata`.

    `backend` is "rapids" (GPU via rapids-singlecell) or "scanpy"
    (CPU via standard scanpy). The rapids path needs `adata.obsm[emb_key]`
    on the GPU (cupy ndarray); we move it transparently and restore the
    numpy view afterwards so downstream `sc.pl.umap` calls (which expect
    numpy) stay happy.
    """
    if backend == "rapids":
        try:
            _compute_umap_rapids(
                adata,
                emb_key=emb_key,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                key_added=key_added,
            )
            return
        except _GPU_OOM_EXCS as e:
            # CUDA OOM: cupy raises MemoryError, cuml/rmm sometimes wraps
            # in RuntimeError. The parent torch process may still be
            # holding a CUDA context that doesn't release fast enough,
            # the LSF GPU may be shared / undersized, or rapids' RMM
            # pool fragmented. Fall back to scanpy CPU rather than
            # killing the whole pipeline — matters most on the LSF path
            # where many jobs run unattended overnight.
            print(
                f"  [{emb_key}] rapids GPU UMAP failed with "
                f"{type(e).__name__}: {e}\n"
                f"    -> falling back to scanpy CPU UMAP for this latent."
            )
            # Best-effort GPU cleanup so subsequent latents may still
            # succeed on the GPU. cupy's default pool caches freed
            # blocks; release them so other processes / iterations get
            # the memory back. RMM has its own pool but cupy's pool is
            # what the failed allocation was using.
            try:
                import cupy as _cp
                _cp.get_default_memory_pool().free_all_blocks()
                _cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:  # noqa: BLE001
                pass
            # Fall through to the scanpy path below.

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=emb_key,
        n_pcs=n_pcs,
        key_added=key_added,
    )
    sc.tl.umap(adata, neighbors_key=key_added)


# CUDA-OOM-ish exception classes we want to catch + fall back from.
# `MemoryError` covers cupy + rmm; `RuntimeError` covers cuml's wrapped
# allocator failures. Wider catches happen in the call site.
_GPU_OOM_EXCS: tuple = (MemoryError, RuntimeError)


def _compute_umap_rapids(
        adata: ad.AnnData,
        emb_key: str,
        n_neighbors: int,
        n_pcs: Optional[int],
        key_added: str,
    ) -> None:
    """rapids-singlecell path — stages the embedding on GPU, runs cuML
    cuKNN + cuML UMAP. May raise MemoryError / RuntimeError on OOM; the
    caller (`_compute_umap`) catches those and falls back to scanpy."""
    import cupy as cp
    import rapids_singlecell as rsc

    # Stage the embedding on GPU. Keep the original numpy view so we
    # can restore it after — `sc.pl.umap` uses the embedding for the
    # neighbours-key fallback only, but other code paths in this
    # script read `obsm` directly (numpy is faster for matplotlib).
    np_emb = np.asarray(adata.obsm[emb_key])
    adata.obsm[emb_key] = cp.asarray(np_emb)
    try:
        rsc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=emb_key,
            n_pcs=n_pcs,
            key_added=key_added,
        )
        rsc.tl.umap(adata, neighbors_key=key_added)
        # `tl.umap` writes the result to `adata.obsm["X_umap"]` as a
        # cupy array; convert back so matplotlib + AnnData I/O work.
        if "X_umap" in adata.obsm and hasattr(adata.obsm["X_umap"], "get"):
            adata.obsm["X_umap"] = adata.obsm["X_umap"].get()
    finally:
        adata.obsm[emb_key] = np_emb


def _save_dual_current(out_path, **savefig_kwargs) -> None:
    """Save the CURRENT matplotlib figure (`plt.gcf()`) as both `.png`
    and `.svg` siblings sharing the same stem. The `format` kwarg is
    derived per-extension; passing one in `savefig_kwargs` would conflict."""
    out_path = Path(out_path)
    savefig_kwargs.pop("format", None)
    for ext in (".png", ".svg"):
        plt.savefig(out_path.with_suffix(ext), format=ext.lstrip("."),
                    **savefig_kwargs)


# ---------------------------------------------------------------------------
# Categorical-cast helper
# ---------------------------------------------------------------------------

def _ensure_categorical(adata: ad.AnnData, key: str) -> None:
    """Force `adata.obs[key]` to be a pandas.Categorical so `sc.pl.umap`
    renders it as DISCRETE (legend, distinct colours per category)
    rather than CONTINUOUS (colourbar). Idempotent — already-categorical
    columns are left alone; numeric columns (e.g. integer-encoded
    `adata_batch_id`) are cast via `astype(str).astype("category")` so
    every label flows through scanpy's categorical code path."""
    if key not in adata.obs.columns:
        return
    import pandas as pd
    col = adata.obs[key]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return
    adata.obs[key] = col.astype(str).astype("category")


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

def make_large_cmap(n: int) -> List:
    """Return a list of `n` distinct colors. Stitches tab20/20b/20c for >20."""
    if n <= 20:
        return list(plt.get_cmap("tab20").colors[:n])
    palettes = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        palettes += list(plt.get_cmap(cmap_name).colors)
    if n <= len(palettes):
        return palettes[:n]
    # Fall back to hsv for >60 categories.
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


# ---------------------------------------------------------------------------
# Per-latent UMAP
# ---------------------------------------------------------------------------

def plot_umap_for_latent(
        adata: ad.AnnData,
        emb_key: str,
        batch_key: str,
        label_keys: List[str],
        out_dir: Path,
        n_neighbors: int = 15,
        n_pcs: Optional[int] = None,
        rasterize: bool = True,
        point_size: Optional[float] = None,
        backend: str = "scanpy",
    ) -> None:
    """Build neighbors + UMAP for one latent, then plot a panel per color key."""
    if emb_key not in adata.obsm:
        print(f"  [{emb_key}] not in obsm; skipping")
        return

    # Each latent gets its OWN neighbours / UMAP graph (under `key_added`).
    print(f"  [{emb_key}] computing kNN ({n_neighbors}) + UMAP "
          f"[backend={backend}] ...")
    _compute_umap(
        adata,
        emb_key=emb_key,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        key_added=emb_key,
        backend=backend,
    )

    n_pcs_string = f"_{n_pcs}_pcs" if n_pcs else ""
    extra_kw = {"size": point_size} if point_size is not None else {}

    # ---- panel 1: color by batch ------------------------------------------
    if batch_key in adata.obs.columns:
        # Cast batch_key to Categorical so sc.pl.umap renders it discrete
        # (legend) rather than continuous (colorbar). adata_batch_id is
        # typically int — without this cast scanpy gives a colorbar from
        # 0..max_batch_id.
        _ensure_categorical(adata, batch_key)
        n_cats = adata.obs[batch_key].nunique()
        palette = make_large_cmap(n_cats)
        out_path = out_dir / f"{emb_key}_umap_by_{batch_key}{n_pcs_string}.svg"
        sc.pl.umap(
            adata,
            color=batch_key,
            palette=palette,
            title=f"{emb_key} — {batch_key}",
            legend_loc="right margin",
            legend_fontsize=10,
            show=False,
            **extra_kw,
        )
        if rasterize:
            for collection in plt.gca().collections:
                collection.set_rasterized(True)
        _save_dual_current(out_path, bbox_inches="tight")
        plt.close()
        print(f"  -> wrote {out_path}")
    else:
        print(f"  [{emb_key}] batch_key '{batch_key}' missing from obs; skipping batch panel")

    # ---- panel(s) by label keys -------------------------------------------
    for label_key in label_keys:
        if label_key not in adata.obs.columns:
            print(f"  [{emb_key}] label '{label_key}' missing from obs; skipping")
            continue
        # Same categorical cast as the batch panel — guarantees every
        # label is rendered discrete regardless of its original dtype.
        _ensure_categorical(adata, label_key)
        n_cats = adata.obs[label_key].nunique()
        palette = make_large_cmap(n_cats)
        out_path = out_dir / f"{emb_key}_umap_by_{label_key}{n_pcs_string}.svg"
        sc.pl.umap(
            adata,
            color=label_key,
            palette=palette,
            title=f"{emb_key} — {label_key}",
            show=False,
            **extra_kw,
        )
        if rasterize:
            for collection in plt.gca().collections:
                collection.set_rasterized(True)
        _save_dual_current(out_path, bbox_inches="tight")
        plt.close()
        print(f"  -> wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--predicted-adata", type=str, required=True,
                    help="Path to predicted_adata.h5ad written by --predict.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: <predicted-adata-dir>/umap_plots).")
    ap.add_argument("--emb-keys", type=str,
                    default="cell_emb,neighborhood_emb,cell_latent,neighborhood_latent,"
                            "X_squint,X_squint_quantized",
                    help="Comma-separated obsm keys to UMAP. Missing keys are skipped.")
    ap.add_argument("--batch-key", type=str, default="adata_batch_id",
                    help="Obs column for the batch panel (default: adata_batch_id).")
    ap.add_argument("--batch-rename", type=str, default=None,
                    help="If given, comma-separated category labels in sorted "
                         "order to remap the batch values. Example: "
                         "'MERFISH,STARmap PLUS' for two-batch datasets.")
    ap.add_argument("--label-keys", type=str,
                    default="cell_type,annotation,niche,spatial_cluster",
                    help="Comma-separated obs columns to color UMAP by, in "
                         "addition to the batch panel. Defaults cover "
                         "multiple datasets — missing columns are silently "
                         "skipped: cell_type (mmb / chl59), annotation "
                         "(spatch_{ov,hcc,coad}_1p), niche (chl59), "
                         "spatial_cluster (spatch_*_1p).")
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="kNN size for sc.pp.neighbors (default 15).")
    ap.add_argument("--n-pcs", type=int, default=None,
                    help="Number of PCs to use (default: None = use embedding directly).")
    ap.add_argument("--point-size", type=float, default=None,
                    help="Scatter point size (default: scanpy auto).")
    ap.add_argument("--no-rasterize", action="store_true",
                    help="Don't rasterize the scatter (vector by default; rasterize "
                         "is recommended for files with >100k points).")
    backend_grp = ap.add_mutually_exclusive_group()
    backend_grp.add_argument(
        "--use-gpu", action="store_true",
        help="Force GPU-accelerated UMAP via rapids-singlecell. Errors "
             "out if rapids-singlecell or cupy isn't importable, or no "
             "GPU is visible. Default: auto-detect (use GPU if available, "
             "else CPU).",
    )
    backend_grp.add_argument(
        "--cpu-only", action="store_true",
        help="Force scanpy CPU UMAP even if a GPU is available. Default: "
             "auto-detect.",
    )
    args = ap.parse_args()

    adata_path = Path(args.predicted_adata)
    adata = ad.read_h5ad(adata_path)
    print(f"Loaded {adata_path}")
    print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    out_dir = Path(args.out_dir) if args.out_dir else adata_path.parent / "umap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_keys = [k.strip() for k in args.emb_keys.split(",") if k.strip()]
    label_keys = [k.strip() for k in args.label_keys.split(",") if k.strip()]

    # ---- batch-rename (categorical relabel) -------------------------------
    if args.batch_rename and args.batch_key in adata.obs.columns:
        # Make sure it's a categorical so .cat.categories works
        adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
        cats = sorted(adata.obs[args.batch_key].cat.categories.tolist())
        new_names = [s.strip() for s in args.batch_rename.split(",")]
        if len(new_names) != len(cats):
            print(f"WARN: --batch-rename has {len(new_names)} names but obs[{args.batch_key}] "
                  f"has {len(cats)} categories ({cats}); skipping rename.")
        else:
            rename = dict(zip(cats, new_names))
            adata.obs[args.batch_key] = (
                adata.obs[args.batch_key].map(rename).astype("category")
            )
            print(f"  batch rename: {rename}")

    # Resolve UMAP backend (rapids-singlecell GPU vs scanpy CPU).
    if args.cpu_only:
        backend = "scanpy"
    elif args.use_gpu:
        backend = _detect_gpu_umap_backend()
        if backend != "rapids":
            raise SystemExit(
                "--use-gpu requested but rapids-singlecell + cupy + a "
                "visible GPU were not all available. Either install "
                "rapids-singlecell or drop --use-gpu (auto-detect falls "
                "back to scanpy CPU)."
            )
    else:
        backend = _detect_gpu_umap_backend()
    print(f"UMAP backend: {backend}")

    print(f"\nLatents to plot: {emb_keys}")
    print(f"Color keys: batch={args.batch_key!r} + labels={label_keys}\n")

    for emb_key in emb_keys:
        print(f"=== {emb_key} ===")
        plot_umap_for_latent(
            adata,
            emb_key=emb_key,
            batch_key=args.batch_key,
            label_keys=label_keys,
            out_dir=out_dir,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
            rasterize=not args.no_rasterize,
            point_size=args.point_size,
            backend=backend,
        )

    print()
    print("=" * 78)
    print(f"Done. Wrote UMAP figures to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
