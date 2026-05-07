"""
Sanity-check plot: spatially variable genes — ground truth vs reconstruction.

Strategy
--------
1. Load the predicted AnnData written by `run_squint.py --predict`.
2. Build an 8-NN spatial graph PER BATCH (the two batches live at unrelated
   coordinate frames).
3. Rank genes by Moran's I on log1p(X). Moran's I measures spatial
   autocorrelation: genes with high I form clear spatial patterns
   (gradients, layers, regions); genes near 0 are spatially uniform.
4. Plot a grid for each batch:  rows = top-N genes, cols = panels selected
   by `--compare`:
     "cell" -> [X (cell raw),  X_hat (cell recon)]
     "nbr"  -> [X_nbr (nbr raw), X_hat_nbr (nbr recon)]
     "both" -> [X, X_hat, X_nbr, X_hat_nbr]   (default)
   Each panel is a scatter at adata.obsm["spatial"] coloured by the gene's
   expression value (log1p) at that cell.

Why this is a good second sanity check
--------------------------------------
- Cell-wise Pearson is dominated by zeros and can be high even when the
  spatial structure is washed out. Moran's-I-ranked side-by-side spatial
  plots show exactly the thing you care about: does the model preserve
  layered / gradient / domain structure across the section?
- Comparing the cell pair (X, X_hat) tells you whether per-cell counts are
  faithfully reconstructed; comparing the neighbourhood pair (X_nbr,
  X_hat_nbr) tells you whether the model captures the smoothed niche-level
  signal. They tell different stories: a model can do well on one and
  poorly on the other.
- Mild blur in the reconstruction is expected (tight bottleneck + 1-hop
  NB loss). What would be a problem: the GT shows a sharp pattern and
  the recon shows a uniform smear or a different pattern.

Storage layout (set by `run_squint.py --predict`):
    adata.X                  -> cell raw counts (sparse)
    adata.layers["X_hat"]    -> cell reconstruction (NB rate * read depth)
    adata.uns["X_nbr"]       -> 1-hop neighborhood mean of raw counts
                                (torch tensor, n_cells x n_genes)
    adata.layers["X_hat_nbr"] -> 1-hop neighborhood mean of cell reconstruction

Usage
-----
    python examples/plot_svg_reconstruction.py \\
        --predicted-adata <ARTIFACTS_DIR>/inference/<run>/predicted_adata.h5ad \\
        --top-n 9 \\
        --out-dir <ARTIFACTS_DIR>/inference/<run>/svg_plots/

Useful flags:
    --svg-source-batch 0   # rank SVGs on batch 0, plot the same genes in
                           # both batches (better for cross-batch comparison).
    --n-perms 0            # skip permutation p-values (faster); ranking by
                           # Moran's I statistic alone is fine for a sanity
                           # check.
    --compare cell         # cell-only 2-column comparison
    --compare nbr          # nbr-only  2-column comparison
    --compare both         # 4-column comparison (default)
"""

import argparse
import warnings
from pathlib import Path

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
import scipy.sparse as sp
import squidpy as sq


def _save_dual(fig, out_path, **savefig_kwargs) -> None:
    """Save the figure as BOTH `.png` and `.svg` (sibling files sharing
    the same stem)."""
    out_path = Path(out_path)
    for ext in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(ext), **savefig_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_log1p_layer(adata: ad.AnnData, layer_name: str = "log1p_X") -> None:
    """Add a log1p-transformed copy of adata.X under adata.layers[layer_name]."""
    if layer_name in adata.layers:
        return
    X = adata.X
    if sp.issparse(X):
        X_log = X.copy().astype(np.float32)
        X_log.data = np.log1p(X_log.data)
    else:
        X_log = np.log1p(np.asarray(X, dtype=np.float32))
    adata.layers[layer_name] = X_log


def _moran_top_genes(
    adata: ad.AnnData,
    top_n: int,
    n_perms: int | None = None,
    n_neighs: int = 8,
    seed: int = 0,
) -> list[str]:
    """
    Return the top-N gene names ranked by Moran's I on log1p(X).
    Builds an 8-NN spatial graph if not already present.

    `n_perms=None` skips permutation p-values (Moran's I statistic alone is
    enough for ranking). squidpy rejects `n_perms=0` with `_assert_positive`,
    so we explicitly pass None instead.
    """
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(
            adata, coord_type="generic", n_neighs=n_neighs
        )
    _ensure_log1p_layer(adata)

    # squidpy: n_perms must be a positive int OR None (None = no permutations).
    perm_kw: dict = {} if n_perms is None else {"n_perms": int(n_perms)}

    sq.gr.spatial_autocorr(
        adata,
        connectivity_key="spatial_connectivities",
        layer="log1p_X",
        mode="moran",
        seed=seed,
        **perm_kw,
    )
    moranI = adata.uns["moranI"]  # already sorted by I desc by squidpy
    return moranI.head(top_n).index.tolist()


def _gene_column(matrix, gene_id, var_index) -> np.ndarray:
    """
    Extract a single gene's per-cell vector from a (sparse, dense, or torch)
    matrix.  `gene_id` may be a gene name (looked up in `var_index`) or an
    int-like string (used as a column position).

    Handles three storage formats:
      - scipy sparse (e.g. adata.X CSR)            -> .toarray()
      - dense numpy array (e.g. adata.layers[..])  -> direct
      - torch.Tensor (e.g. adata.uns['X_nbr'])     -> .cpu().numpy()
    """
    if gene_id in var_index:
        j = var_index.get_loc(gene_id)
    else:
        try:
            j = int(gene_id)
        except (TypeError, ValueError) as e:
            raise KeyError(
                f"Gene '{gene_id}' not in var.index and not an integer position."
            ) from e
    col = matrix[:, j]
    # torch tensor ?  detach -> cpu -> numpy
    if hasattr(col, "detach") and hasattr(col, "cpu"):
        col = col.detach().cpu().numpy()
    elif hasattr(col, "toarray"):
        col = col.toarray().ravel()
    return np.asarray(col, dtype=np.float32).ravel()


def _spatial_panel(
    ax,
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    point_size: float = 0.5,
) -> None:
    """Scatter cells at xy, coloured by `values`. Robust 2-98 percentile clip."""
    if values.size == 0 or np.all(values == values[0]):
        # Constant-valued column (dead gene); just plot uniform.
        sc_h = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=point_size,
                          cmap=cmap, linewidths=0, rasterized=True)
    else:
        lo, hi = np.percentile(values, [2, 98])
        if lo == hi:
            lo, hi = float(values.min()), float(values.max() + 1e-6)
        sc_h = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=np.clip(values, lo, hi),
            s=point_size, cmap=cmap,
            vmin=lo, vmax=hi,
            linewidths=0, rasterized=True,
        )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc_h, ax=ax, fraction=0.046, pad=0.02)


# ---------------------------------------------------------------------------
# Matrix-source resolution
# ---------------------------------------------------------------------------

def _resolve_source(adata: ad.AnnData, name: str):
    """
    Return the (n_cells, n_genes) matrix associated with `name`, looking it up
    across the storage slots used by the SQUINT predict step:

        - "X"          -> adata.X                       (cell ground truth)
        - "X_hat"      -> adata.layers["X_hat"]         (cell reconstruction)
        - "X_nbr"      -> adata.layers["X_nbr"]  *or*   (nbr ground truth)
                         adata.uns["X_nbr"]             (older predict outputs)
        - "X_hat_nbr"  -> adata.layers["X_hat_nbr"]     (nbr reconstruction)

    Raises a clear error when the requested source is missing.
    """
    if name == "X":
        return adata.X
    if name in adata.layers:
        return adata.layers[name]
    if name in adata.uns:
        return adata.uns[name]
    raise KeyError(
        f"Cannot find '{name}' in adata. Looked at adata.layers={list(adata.layers.keys())} "
        f"and adata.uns keys={list(adata.uns.keys())}."
    )


def _ensure_X_nbr_for_batch(
        adata_b: ad.AnnData,
        n_neighs: int = 8,
    ) -> None:
    """
    Make sure `adata_b.layers["X_nbr"]` is populated so the nbr ground-truth
    panels can be plotted. Three sources, in priority order:

      1. Already in `adata_b.layers["X_nbr"]` — newer predict outputs.
      2. In `adata_b.uns["X_nbr"]` (torch tensor or array) — older predict
         outputs that stashed it in `.uns`. Migrated into `.layers`.
      3. Otherwise: compute on the fly from `adata_b.X` using an 8-NN spatial
         graph + self-loop, matching the training-pipeline graph
         (`sq.gr.spatial_neighbors(set_diag=True, n_neighs=8)`) so the result
         is identical to what the model trained against.

    The on-the-fly path is what's required for predicted_adata.h5ad files
    written before this script started writing X_nbr to layers.
    """
    if "X_nbr" in adata_b.layers:
        return

    # (2) Migrate from .uns if present.
    if "X_nbr" in adata_b.uns:
        x_nbr_arr = adata_b.uns["X_nbr"]
        if hasattr(x_nbr_arr, "detach"):       # torch.Tensor
            x_nbr_arr = x_nbr_arr.detach().cpu().numpy()
        adata_b.layers["X_nbr"] = np.asarray(x_nbr_arr, dtype=np.float32)
        return

    # (3) Build it from the spatial graph.  Use set_diag=True so each cell
    #     is counted in its own neighbourhood — matches training semantics.
    sq.gr.spatial_neighbors(
        adata_b,
        coord_type="generic",
        n_neighs=n_neighs,
        set_diag=True,
    )
    A = adata_b.obsp["spatial_connectivities"]   # (N, N) sparse 0/1 with self-loops
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    # Row-normalise the adjacency so (A_norm @ X) yields the per-cell mean.
    D_inv  = sp.diags(1.0 / deg)
    A_norm = D_inv @ A
    X = adata_b.X
    X_nbr = A_norm @ X
    if sp.issparse(X_nbr):
        X_nbr = X_nbr.toarray()
    adata_b.layers["X_nbr"] = np.asarray(X_nbr, dtype=np.float32)


# Column specs for each --compare mode.  Each entry is (storage_name, panel_title_template).
# `{gene}` is substituted at plot time.
_COMPARE_LAYOUTS = {
    "cell": [("X",     "{gene}: cell raw (log1p)"),
             ("X_hat", "{gene}: cell recon (log1p)")],
    "nbr":  [("X_nbr",     "{gene}: nbr raw (log1p)"),
             ("X_hat_nbr", "{gene}: nbr recon (log1p)")],
    "both": [("X",         "{gene}: cell raw (log1p)"),
             ("X_hat",     "{gene}: cell recon (log1p)"),
             ("X_nbr",     "{gene}: nbr raw (log1p)"),
             ("X_hat_nbr", "{gene}: nbr recon (log1p)")],
}


# ---------------------------------------------------------------------------
# Per-batch plotting
# ---------------------------------------------------------------------------

def plot_batch(
    adata_b: ad.AnnData,
    gene_list: list[str],
    out_path: Path,
    compare: str = "both",
    log1p: bool = True,
    n_neighs: int = 8,
) -> None:
    """
    For one batch's AnnData (already subset), plot top-N genes as a grid:
        rows = genes, cols = panels per `compare` mode.

    compare:
        "cell" -> 2 cols [X, X_hat]
        "nbr"  -> 2 cols [X_nbr, X_hat_nbr]
        "both" -> 4 cols [X, X_hat, X_nbr, X_hat_nbr]

    If the chosen mode needs X_nbr but it isn't in the AnnData, it is
    computed on the fly from the spatial graph (8-NN + self-loop, matching
    the training-pipeline aggregation).
    """
    if compare not in _COMPARE_LAYOUTS:
        raise ValueError(f"compare must be one of {list(_COMPARE_LAYOUTS)}, got {compare!r}")
    layout = _COMPARE_LAYOUTS[compare]
    needed = {name for name, _ in layout}
    if "X_nbr" in needed:
        _ensure_X_nbr_for_batch(adata_b, n_neighs=n_neighs)

    n_cols = len(layout)
    n_rows = len(gene_list)

    # ~4 inches per panel column keeps panels square-ish at 3 in row height.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    xy = adata_b.obsm["spatial"]
    var_index = adata_b.var.index

    # Pre-resolve matrices once (rather than per gene * column).
    sources = {name: _resolve_source(adata_b, name) for name, _ in layout}

    for r, gene in enumerate(gene_list):
        for c, (name, title_tpl) in enumerate(layout):
            vec = _gene_column(sources[name], gene, var_index)
            if log1p:
                vec = np.log1p(np.clip(vec, 0, None))
            _spatial_panel(axes[r, c], xy, vec, title_tpl.format(gene=gene))

    fig.tight_layout()
    _save_dual(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predicted-adata", type=str, required=True,
                    help="Path to predicted_adata.h5ad written by --predict.")
    ap.add_argument("--top-n", type=int, default=9,
                    help="Number of top spatially-variable genes to plot per batch.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Where to save figures (default: <predicted-adata-dir>/svg_plots).")
    ap.add_argument("--n-perms", type=int, default=0,
                    help="Permutations for Moran's I p-value. 0 = skip (faster). "
                         "Internally we pass None to squidpy in the skip case "
                         "(squidpy rejects n_perms=0).")
    ap.add_argument("--n-neighs", type=int, default=8,
                    help="Number of spatial neighbours for Moran's I graph.")
    ap.add_argument("--svg-source-batch", type=int, default=None,
                    help="If set, rank SVGs on this batch and plot the same "
                         "genes for every batch (cross-batch comparison). "
                         "Default: rank per batch independently.")
    ap.add_argument("--compare", type=str, default="both",
                    choices=["cell", "nbr", "both"],
                    help="Which panels to plot per gene: "
                         "'cell' = [X, X_hat] (2 cols); "
                         "'nbr' = [X_nbr, X_hat_nbr] (2 cols); "
                         "'both' = all four (4 cols, default).")
    args = ap.parse_args()

    adata_path = Path(args.predicted_adata)
    adata = ad.read_h5ad(adata_path)
    print(f"Loaded {adata_path}")
    print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"  layers={list(adata.layers.keys())}")
    print(f"  obs columns={list(adata.obs.columns)}")

    # ---- sanity: required slots --------------------------------------------
    # Determine which sources are needed for the chosen compare mode. X_nbr
    # is special: if it's absent from disk we'll compute it on the fly from
    # the spatial graph inside `plot_batch`. Everything else (X, X_hat,
    # X_hat_nbr) must already be present.
    needed = {name for name, _ in _COMPARE_LAYOUTS[args.compare]}
    for name in needed:
        if name == "X_nbr":
            continue   # auto-computed on the fly when missing
        try:
            _resolve_source(adata, name)
        except KeyError as e:
            raise SystemExit(
                f"--compare={args.compare} requires '{name}' in the AnnData. {e}"
            ) from e
    if "spatial" not in adata.obsm:
        raise SystemExit("Expected adata.obsm['spatial'] to be present.")
    if "adata_batch_id" not in adata.obs.columns:
        raise SystemExit("Expected adata.obs['adata_batch_id'] to be present.")

    out_dir = Path(args.out_dir) if args.out_dir else adata_path.parent / "svg_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_ids = sorted(int(b) for b in adata.obs["adata_batch_id"].unique())
    print(f"Batches present: {batch_ids}")

    # ---- optional: pre-rank SVGs on a single source batch -------------------
    fixed_genes = None
    if args.svg_source_batch is not None:
        b = args.svg_source_batch
        adata_b = adata[adata.obs["adata_batch_id"] == b].copy()
        print(f"Computing Moran's I on batch {b} (source for SVG ranking)...")
        fixed_genes = _moran_top_genes(
            adata_b, args.top_n, n_perms=(args.n_perms or None), n_neighs=args.n_neighs,
        )
        print(f"Top {args.top_n} SVGs (batch {b}):")
        for g in fixed_genes:
            print(f"  - {g}")

    # ---- per-batch plot ------------------------------------------------------
    for b in batch_ids:
        adata_b = adata[adata.obs["adata_batch_id"] == b].copy()

        if fixed_genes is None:
            print(f"Computing Moran's I on batch {b}...")
            genes = _moran_top_genes(
                adata_b, args.top_n, n_perms=(args.n_perms or None), n_neighs=args.n_neighs,
            )
            print(f"Top {args.top_n} SVGs (batch {b}):")
            for g in genes:
                print(f"  - {g}")
        else:
            genes = fixed_genes

        out_path = out_dir / f"svg_recon_batch{b}_compare-{args.compare}.png"
        plot_batch(adata_b, genes, out_path, compare=args.compare,
                   n_neighs=args.n_neighs)
        print(f"  -> wrote {out_path}")

    print()
    print("=" * 78)
    print(f"Done. Wrote {len(batch_ids)} figure(s) to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
