"""
Spatial plot of VQ code assignments — one panel per tissue section per level,
per codebook.

Strategy
--------
1. Load the predicted AnnData written by `run_squint.py --predict`.
2. Detect the codebook structure. Three layouts are supported:
     - VQNiche_Dual: TWO codebooks, addressed under per-branch keys
         adata.obs['cell_code_index']           or adata.obsm['cell_code_indices']
         adata.obs['neighborhood_code_index']   or adata.obsm['neighborhood_code_indices']
     - Legacy single-codebook (VQNiche):
         adata.obs['code_index']                or adata.obsm['code_indices']
   1D = single-level codebook; 2D = multi-level (RVQ / ConditionalVQ /
   multi-head). Level labels disambiguate "level k" (hierarchical) vs
   "head k" (multi-head) using adata.uns['squint'].
3. For each tissue section (`adata.obs['adata_batch_id']`) and each codebook,
   render a spatial scatter at `adata.obsm['spatial']` coloured by per-cell
   code id.
   - Small codebooks (<= 30 codes): tab20-style categorical palette + legend.
   - Larger codebooks (> 30 codes):  cyclic hsv cmap with colorbar (no legend).

Output (per batch):
    code_index_batch{b}_{codebook}_{level}.png
        one-panel figure per codebook per level
    code_index_batch{b}_{codebook}_summary.png
        multi-level summary (only emitted when the codebook has >1 level)
    code_index_batch{b}_codebooks.png
        cross-codebook dual view: cell + neighborhood side-by-side at the
        primary level (only emitted when >=2 codebooks are present)
For dual outputs, the cross-codebook figure is the most useful diagnostic:
the cell codebook should look mosaic / cell-type-shaped while the
neighborhood codebook should look like coherent spatial domains.

Usage
-----
    python examples/plot_code_indices_spatial.py \\
        --predicted-adata <ARTIFACTS_DIR>/inference/<run>/predicted_adata.h5ad \\
        --out-dir <ARTIFACTS_DIR>/inference/<run>/code_index_plots/

Useful flags:
    --point-size 1.5                   # scatter point size
    --max-categorical 30               # codebooks larger than this use a
                                       # continuous colourmap (no legend)
    --shared-axes                      # equalize x/y limits across batches
                                       # (only meaningful if both sections
                                       # live in the same coordinate frame)
    --max-codes-in-legend 30           # cap legend entries (skip if exceeds)
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


def _save_dual(fig, out_path, **savefig_kwargs) -> None:
    """Save the figure as BOTH `.png` and `.svg` (sibling files sharing
    the same stem). out_path's suffix is ignored — both extensions are
    always written. Keeps every plot inspectable as a raster preview AND
    as a vector source for figures."""
    out_path = Path(out_path)
    for ext in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(ext), **savefig_kwargs)


# ---------------------------------------------------------------------------
# Code-index extraction
# ---------------------------------------------------------------------------

def _extract_one_codebook(
        adata: ad.AnnData,
        prefix: str,
        n_quant_meta_key: str,
        n_heads_meta_key: str,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Pull a single codebook's per-cell indices out of the AnnData, using the
    convention:
        adata.obs[f"{prefix}_code_index"]    (1D, n_cells,)              for single-level
        adata.obsm[f"{prefix}_code_indices"] (2D, n_cells x n_levels)    for multi-level
    Empty `prefix` falls back to the legacy keys `code_index` / `code_indices`.

    Returns (codes_2d, level_labels) or None if neither slot is populated.
    Level labels disambiguate hierarchical levels ("level 1", "level 2") vs
    multi-head ("head 1", "head 2") via `adata.uns['squint']`.
    """
    obs_key  = f"{prefix}_code_index"   if prefix else "code_index"
    obsm_key = f"{prefix}_code_indices" if prefix else "code_indices"

    if obsm_key in adata.obsm:
        codes_2d = np.asarray(adata.obsm[obsm_key]).astype(int)
        if codes_2d.ndim == 1:
            codes_2d = codes_2d[:, None]
    elif obs_key in adata.obs.columns:
        codes_2d = np.asarray(adata.obs[obs_key].values, dtype=int)[:, None]
    else:
        return None

    n_levels = codes_2d.shape[1]
    squint_meta = dict(adata.uns.get("squint", {}))
    n_quant = int(squint_meta.get(n_quant_meta_key, 1) or 1)
    n_heads = int(squint_meta.get(n_heads_meta_key, 1) or 1)

    if n_levels == 1:
        labels = ["code"]
    elif n_quant > 1 and n_quant == n_levels:
        labels = [f"level {q + 1}" for q in range(n_levels)]
    elif n_heads > 1 and n_heads == n_levels:
        labels = [f"head {h + 1}" for h in range(n_levels)]
    else:
        labels = [f"code {k + 1}" for k in range(n_levels)]
    return codes_2d, labels


def _extract_codebooks(adata: ad.AnnData) -> List[dict]:
    """
    Detect every codebook present in the AnnData and return a list of
    codebook descriptors:
        [{"name": "cell",          "codes_2d": ..., "labels": [...]},
         {"name": "neighborhood",  "codes_2d": ..., "labels": [...]}]

    Three layouts are supported:
      1. Dual VQNiche_Dual: per-branch keys
           obs/obsm: cell_code_index(es), neighborhood_code_index(es)
         -> two codebooks: 'cell' + 'neighborhood'
      2. Legacy single-codebook (VQNiche etc.): unsuffixed keys
           obs/obsm: code_index, code_indices
         -> one codebook: 'code'
      3. Mixed (newer dual write that also fills the legacy keys for
         back-compat): only the per-branch keys are reported, the legacy
         alias is ignored to avoid duplicating the niche-branch panel.

    Returns at least one codebook; raises if none are present.
    """
    squint_meta = dict(adata.uns.get("squint", {}))
    found: List[dict] = []

    # --- Dual layout ------------------------------------------------------
    cell = _extract_one_codebook(
        adata,
        prefix="cell",
        n_quant_meta_key="num_quantizers_cell",
        n_heads_meta_key="num_heads",
    )
    nbr = _extract_one_codebook(
        adata,
        prefix="neighborhood",
        n_quant_meta_key="num_quantizers_niche",
        n_heads_meta_key="num_heads",
    )
    if cell is not None:
        found.append({"name": "cell",         "codes_2d": cell[0], "labels": cell[1]})
    if nbr is not None:
        found.append({"name": "neighborhood", "codes_2d": nbr[0],  "labels": nbr[1]})

    # --- Legacy single-codebook fallback ---------------------------------
    if not found:
        legacy = _extract_one_codebook(
            adata,
            prefix="",
            n_quant_meta_key="num_quantizers",
            n_heads_meta_key="num_heads",
        )
        if legacy is None:
            raise SystemExit(
                "No codebook indices found. Expected one of:\n"
                "  - adata.obs['cell_code_index']  / adata.obsm['cell_code_indices']\n"
                "  - adata.obs['neighborhood_code_index'] / adata.obsm['neighborhood_code_indices']\n"
                "  - adata.obs['code_index']      / adata.obsm['code_indices']\n"
                f"obs columns: {list(adata.obs.columns)}\n"
                f"obsm keys:   {list(adata.obsm.keys())}"
            )
        found.append({"name": "code", "codes_2d": legacy[0], "labels": legacy[1]})

    return found


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _build_palette(num_codes: int, max_categorical: int = 30) -> Tuple[ListedColormap, bool]:
    """
    Pick a colormap appropriate for the number of distinct codes.

    Returns (cmap, is_categorical). Categorical = high-contrast palette with
    one distinct colour per code (suitable for legends). Non-categorical =
    cyclic / continuous palette (used when the codebook is too large to
    enumerate in a legend).
    """
    if num_codes <= max_categorical:
        # tab20 has 20 distinct colours; for 21..30 we tile tab20 + tab20b.
        if num_codes <= 20:
            cmap = plt.get_cmap("tab20", num_codes)
        else:
            tab20  = plt.get_cmap("tab20").colors
            tab20b = plt.get_cmap("tab20b").colors
            colors = list(tab20) + list(tab20b)
            cmap = ListedColormap(colors[:num_codes])
        return cmap, True
    # Larger codebooks: cyclic hsv gives high contrast even at 200+ codes.
    return plt.get_cmap("hsv"), False


def _scatter_codes(
        ax,
        xy: np.ndarray,
        codes: np.ndarray,
        title: str,
        point_size: float = 1.5,
        max_categorical: int = 30,
        show_legend: bool = True,
        max_codes_in_legend: int = 30,
    ) -> None:
    """Scatter `xy` (n,2) coloured by integer `codes` (n,)."""
    unique_codes = np.unique(codes)
    n_unique     = unique_codes.size
    cmap, is_categorical = _build_palette(int(codes.max()) + 1, max_categorical)

    if is_categorical:
        # Map raw code id -> contiguous palette index for ListedColormap
        # (so colours stay stable even when some codes are unused in this
        # batch). For the colour, we use the raw code id modulo cmap.N to
        # keep code-1 -> color-1 globally — easier to compare across batches.
        c = codes % cmap.N
        sc_h = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=c, cmap=cmap,
            vmin=0, vmax=cmap.N - 1,
            s=point_size, linewidths=0, rasterized=True,
        )
        if show_legend and n_unique <= max_codes_in_legend:
            handles = [
                Line2D([0], [0], marker='o', linestyle='',
                       markersize=5,
                       color=cmap(int(uc) % cmap.N),
                       label=str(int(uc)))
                for uc in unique_codes
            ]
            ax.legend(
                handles=handles,
                title="code",
                bbox_to_anchor=(1.02, 1.0), loc="upper left",
                fontsize=6, title_fontsize=7,
                ncol=max(1, n_unique // 15),
                handletextpad=0.3, labelspacing=0.2, borderaxespad=0.0,
                frameon=False,
            )
    else:
        # Continuous-ish colormap, no legend. Scale codes to [0, 1] for hsv.
        max_code = max(int(codes.max()), 1)
        c = codes / float(max_code)
        sc_h = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=c, cmap=cmap, vmin=0, vmax=1,
            s=point_size, linewidths=0, rasterized=True,
        )
        cbar = plt.colorbar(sc_h, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(f"code id / {max_code}", fontsize=6)
        cbar.ax.tick_params(labelsize=6)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Per-batch plotting
# ---------------------------------------------------------------------------

def plot_one_panel(
        adata_b: ad.AnnData,
        codes_b: np.ndarray,
        level_label: str,
        out_path: Path,
        point_size: float = 1.5,
        max_categorical: int = 30,
        max_codes_in_legend: int = 30,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> None:
    """One-panel figure: spatial scatter of one level's codes for one batch."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    title = f"batch={adata_b.uns.get('_plot_batch_label', '?')}  ({level_label})  " \
            f"unique={int(np.unique(codes_b).size)}"
    _scatter_codes(
        ax,
        xy=adata_b.obsm["spatial"],
        codes=codes_b,
        title=title,
        point_size=point_size,
        max_categorical=max_categorical,
        max_codes_in_legend=max_codes_in_legend,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    _save_dual(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_batch_summary(
        adata_b: ad.AnnData,
        codes_2d: np.ndarray,
        level_labels: List[str],
        out_path: Path,
        point_size: float = 1.5,
        max_categorical: int = 30,
        max_codes_in_legend: int = 30,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> None:
    """One figure per batch with all levels side-by-side as subplots."""
    n_levels = codes_2d.shape[1]
    fig, axes = plt.subplots(
        1, n_levels, figsize=(7 * n_levels, 6), squeeze=False,
    )
    batch_label = adata_b.uns.get("_plot_batch_label", "?")
    for q in range(n_levels):
        codes_q = codes_2d[:, q]
        title = f"batch={batch_label}  ({level_labels[q]})  " \
                f"unique={int(np.unique(codes_q).size)}"
        _scatter_codes(
            axes[0, q],
            xy=adata_b.obsm["spatial"],
            codes=codes_q,
            title=title,
            point_size=point_size,
            max_categorical=max_categorical,
            max_codes_in_legend=max_codes_in_legend,
            # Only show legend on the last column to save horizontal space
            # in multi-level summaries.
            show_legend=(q == n_levels - 1),
        )
        if xlim is not None:
            axes[0, q].set_xlim(*xlim)
        if ylim is not None:
            axes[0, q].set_ylim(*ylim)
    fig.tight_layout()
    _save_dual(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--predicted-adata", type=str, required=True,
                    help="Path to predicted_adata.h5ad written by --predict.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: <predicted-adata-dir>/code_index_plots).")
    ap.add_argument("--point-size", type=float, default=1.5,
                    help="Scatter point size (default: 1.5).")
    ap.add_argument("--max-categorical", type=int, default=30,
                    help="Codebooks with more than this many codes use a "
                         "continuous-style colormap (no legend).")
    ap.add_argument("--max-codes-in-legend", type=int, default=30,
                    help="Cap on legend entries.")
    ap.add_argument("--shared-axes", action="store_true",
                    help="Use the same x/y limits for every batch — only "
                         "meaningful if both sections are in a shared "
                         "coordinate frame.")
    ap.add_argument("--no-summary", action="store_true",
                    help="Skip the multi-panel summary figure.")
    ap.add_argument("--no-per-level", action="store_true",
                    help="Skip per-(batch, level) one-panel figures.")
    args = ap.parse_args()

    adata_path = Path(args.predicted_adata)
    adata = ad.read_h5ad(adata_path)
    print(f"Loaded {adata_path}")
    print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  obsm keys:   {list(adata.obsm.keys())}")
    print(f"  uns squint:  {dict(adata.uns.get('squint', {}))}")

    if "spatial" not in adata.obsm:
        raise SystemExit("Expected adata.obsm['spatial'] to be present.")
    if "adata_batch_id" not in adata.obs.columns:
        raise SystemExit("Expected adata.obs['adata_batch_id'] to be present.")

    out_dir = Path(args.out_dir) if args.out_dir else adata_path.parent / "code_index_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- extract codebook arrays -------------------------------------------
    codebooks = _extract_codebooks(adata)
    print(f"\nFound {len(codebooks)} codebook(s):")
    for cb in codebooks:
        print(f"  [{cb['name']}]  shape={cb['codes_2d'].shape}  levels={cb['labels']}")
        for q, lbl in enumerate(cb['labels']):
            uniq = np.unique(cb['codes_2d'][:, q])
            print(f"    {lbl}: range=[{uniq.min()}, {uniq.max()}]  n_unique={uniq.size}")

    # ---- shared axes (optional) --------------------------------------------
    xlim = ylim = None
    if args.shared_axes:
        xy_all = adata.obsm["spatial"]
        pad_x = 0.02 * (xy_all[:, 0].max() - xy_all[:, 0].min() + 1e-6)
        pad_y = 0.02 * (xy_all[:, 1].max() - xy_all[:, 1].min() + 1e-6)
        xlim = (xy_all[:, 0].min() - pad_x, xy_all[:, 0].max() + pad_x)
        ylim = (xy_all[:, 1].min() - pad_y, xy_all[:, 1].max() + pad_y)

    # ---- per-batch plotting ------------------------------------------------
    batch_ids = sorted(int(b) for b in adata.obs["adata_batch_id"].unique())
    print(f"\nBatches: {batch_ids}")

    for b in batch_ids:
        mask = (adata.obs["adata_batch_id"].values.astype(int) == b)
        adata_b = adata[mask].copy()
        adata_b.uns["_plot_batch_label"] = str(b)

        # ---- per-codebook output ------------------------------------------
        for cb in codebooks:
            cb_name  = cb["name"]
            codes_b  = cb["codes_2d"][mask]                 # (n_in_batch, n_levels)
            cb_labels = cb["labels"]

            # multi-level summary (only when codebook has >1 level)
            if not args.no_summary and len(cb_labels) > 1:
                out_summary = out_dir / f"code_index_batch{b}_{cb_name}_summary.png"
                # Prefix labels with the codebook name so titles are
                # unambiguous in multi-codebook runs.
                titled_labels = [f"{cb_name}: {l}" for l in cb_labels]
                plot_batch_summary(
                    adata_b, codes_b, titled_labels, out_summary,
                    point_size=args.point_size,
                    max_categorical=args.max_categorical,
                    max_codes_in_legend=args.max_codes_in_legend,
                    xlim=xlim, ylim=ylim,
                )
                print(f"  -> wrote {out_summary}")

            # one-panel per (batch, codebook, level)
            if not args.no_per_level:
                for q, lbl in enumerate(cb_labels):
                    slug = lbl.replace(" ", "")
                    out_path = out_dir / f"code_index_batch{b}_{cb_name}_{slug}.png"
                    title_label = f"{cb_name}: {lbl}" if cb_name != "code" else lbl
                    plot_one_panel(
                        adata_b, codes_b[:, q], title_label, out_path,
                        point_size=args.point_size,
                        max_categorical=args.max_categorical,
                        max_codes_in_legend=args.max_codes_in_legend,
                        xlim=xlim, ylim=ylim,
                    )
                    print(f"  -> wrote {out_path}")

        # ---- cross-codebook dual view (only when >=2 codebooks) -----------
        # Lays the *first level* of each codebook side-by-side in one
        # figure so cell-vs-niche disentanglement is immediately visible.
        # Skipped for legacy single-codebook predict outputs.
        if not args.no_summary and len(codebooks) >= 2:
            out_dual = out_dir / f"code_index_batch{b}_codebooks.png"
            n_cb = len(codebooks)
            fig, axes = plt.subplots(1, n_cb,
                                     figsize=(7 * n_cb, 6),
                                     squeeze=False)
            for c, cb in enumerate(codebooks):
                cb_name  = cb["name"]
                codes_b  = cb["codes_2d"][mask][:, 0]   # primary level
                lbl0     = cb["labels"][0]
                title    = (f"batch={b}  codebook={cb_name}  ({lbl0})  "
                            f"unique={int(np.unique(codes_b).size)}")
                _scatter_codes(
                    axes[0, c],
                    xy=adata_b.obsm["spatial"],
                    codes=codes_b,
                    title=title,
                    point_size=args.point_size,
                    max_categorical=args.max_categorical,
                    max_codes_in_legend=args.max_codes_in_legend,
                    show_legend=(c == n_cb - 1),
                )
                if xlim is not None:
                    axes[0, c].set_xlim(*xlim)
                if ylim is not None:
                    axes[0, c].set_ylim(*ylim)
            fig.tight_layout()
            _save_dual(fig, out_dual, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> wrote {out_dual}")

    print()
    print("=" * 78)
    print(f"Done. Wrote figures to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
