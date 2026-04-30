"""
Sanity-check plot: spatially variable genes — ground truth vs reconstruction.

Strategy
--------
1. Load the predicted AnnData written by `run_squint_mmb_smb.py --predict`.
2. Build an 8-NN spatial graph PER BATCH (the two batches live at unrelated
   coordinate frames).
3. Rank genes by Moran's I on log1p(X). Moran's I measures spatial
   autocorrelation: genes with high I form clear spatial patterns
   (gradients, layers, regions); genes near 0 are spatially uniform.
4. Plot a grid for each batch:  rows = top-N genes, cols = [raw X, X_hat],
   each panel a scatter at adata.obsm["spatial"] coloured by gene expression.

Why this is a good second sanity check
--------------------------------------
- Cell-wise Pearson is dominated by zeros and can be high even when the
  spatial structure is washed out. Moran's-I-ranked side-by-side spatial
  plots show exactly the thing you care about: does the model preserve
  layered / gradient / domain structure across the section?
- Mild blur in the reconstruction is expected (tight bottleneck + 1-hop
  NB loss). What would be a problem: the GT shows a sharp pattern and
  X_hat shows a uniform smear or a different pattern.

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
    --layer X_hat_nbr      # plot the 1-hop neighborhood reconstruction
                           # instead of per-cell X_hat.
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import squidpy as sq


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
    Extract a single gene's per-cell vector from a (sparse or dense) matrix.
    `gene_id` may be a gene name (looked up in `var_index`) or an int-like
    string (used as a column position).
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
    if hasattr(col, "toarray"):
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
# Per-batch plotting
# ---------------------------------------------------------------------------

def plot_batch(
    adata_b: ad.AnnData,
    gene_list: list[str],
    out_path: Path,
    recon_layer: str = "X_hat",
    log1p: bool = True,
) -> None:
    """
    For one batch's AnnData (already subset), plot top-N genes:
        rows = genes, cols = [raw X, recon_layer (e.g. X_hat)].
    """
    n = len(gene_list)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n), squeeze=False)

    xy = adata_b.obsm["spatial"]
    var_index = adata_b.var.index

    for r, gene in enumerate(gene_list):
        x_raw = _gene_column(adata_b.X, gene, var_index)
        x_hat = _gene_column(adata_b.layers[recon_layer], gene, var_index)
        if log1p:
            x_raw = np.log1p(np.clip(x_raw, 0, None))
            x_hat = np.log1p(np.clip(x_hat, 0, None))
        _spatial_panel(axes[r, 0], xy, x_raw, f"{gene}: raw (log1p)")
        _spatial_panel(axes[r, 1], xy, x_hat, f"{gene}: {recon_layer} (log1p)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    ap.add_argument("--layer", type=str, default="X_hat",
                    choices=["X_hat", "X_hat_nbr"],
                    help="Which reconstruction layer to plot against the raw X.")
    args = ap.parse_args()

    adata_path = Path(args.predicted_adata)
    adata = ad.read_h5ad(adata_path)
    print(f"Loaded {adata_path}")
    print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"  layers={list(adata.layers.keys())}")
    print(f"  obs columns={list(adata.obs.columns)}")

    # ---- sanity: required slots ---------------------------------------------
    if args.layer not in adata.layers:
        raise SystemExit(
            f"Expected adata.layers['{args.layer}'] to be present; "
            f"have {list(adata.layers.keys())}."
        )
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

        out_path = out_dir / f"svg_recon_batch{b}_{args.layer}.png"
        plot_batch(adata_b, genes, out_path, recon_layer=args.layer)
        print(f"  -> wrote {out_path}")

    print()
    print("=" * 78)
    print(f"Done. Wrote {len(batch_ids)} figure(s) to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
