"""
Spatial plot of VQ code assignments — one panel per tissue section per level.

Strategy
--------
1. Load the predicted AnnData written by `run_squint_mmb_smb.py --predict`.
2. Detect the codebook structure from the AnnData:
     - 1D indices in `adata.obs['code_index']`     -> one level (single-codebook VQ)
     - 2D indices in `adata.obsm['code_indices']`  -> multiple levels:
          * multi-head VQ (heads = num_heads)      — parallel codebooks
          * RVQ / ConditionalVQ (num_quantizers)   — sequential / hierarchical levels
   The label printed on each panel disambiguates "head k" vs "level k" using
   `adata.uns['squint']`.
3. For each tissue section (`adata.obs['adata_batch_id']`) and each code level,
   render a scatter at `adata.obsm['spatial']` coloured by the per-cell code id.
   - Small codebooks (<= 30 codes): tab20-style categorical palette + legend.
   - Larger codebooks (> 30 codes):  high-contrast cyclic cmap (hsv), no legend.

Output: one PNG per (batch, level), plus a multi-panel summary figure per batch.

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
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Code-index extraction
# ---------------------------------------------------------------------------

def _extract_code_levels(
        adata: ad.AnnData,
    ) -> Tuple[np.ndarray, List[str]]:
    """
    Return a 2D array of code ids (n_cells, n_levels) plus a per-level label.

    Sources (in priority order):
      1. `adata.obsm['code_indices']` — multi-level / multi-head case.
         Shape (n_cells, n_levels). One column per level.
      2. `adata.obs['code_index']`    — single-level case.
         Reshaped to (n_cells, 1).

    Labels disambiguate "head" vs "level" using `adata.uns['squint']`:
      - num_quantizers > 1 -> labels = "level 1", "level 2", ...
      - else if num_heads > 1 -> labels = "head 1", "head 2", ...
      - else -> labels = ["code"]
    """
    if "code_indices" in adata.obsm:
        codes_2d = np.asarray(adata.obsm["code_indices"]).astype(int)
        if codes_2d.ndim == 1:
            codes_2d = codes_2d[:, None]
    elif "code_index" in adata.obs.columns:
        codes_2d = np.asarray(adata.obs["code_index"].values, dtype=int)[:, None]
    else:
        raise SystemExit(
            "Neither adata.obs['code_index'] nor adata.obsm['code_indices'] "
            "is present. Was this AnnData written by `--predict`?"
        )

    n_levels = codes_2d.shape[1]
    squint_meta = dict(adata.uns.get("squint", {}))
    n_quant = int(squint_meta.get("num_quantizers", 1) or 1)
    n_heads = int(squint_meta.get("num_heads", 1) or 1)

    if n_levels == 1:
        labels = ["code"]
    elif n_quant > 1 and n_quant == n_levels:
        labels = [f"level {q + 1}" for q in range(n_levels)]
    elif n_heads > 1 and n_heads == n_levels:
        labels = [f"head {h + 1}" for h in range(n_levels)]
    else:
        # Fallback: ambiguous — just call them "code k".
        labels = [f"code {k + 1}" for k in range(n_levels)]
    return codes_2d, labels


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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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

    # ---- extract code arrays ------------------------------------------------
    codes_2d, level_labels = _extract_code_levels(adata)
    n_cells, n_levels = codes_2d.shape
    print(f"\nCodes:  shape={codes_2d.shape}  levels={level_labels}")
    for q, lbl in enumerate(level_labels):
        unique = np.unique(codes_2d[:, q])
        print(f"  {lbl}: range=[{unique.min()}, {unique.max()}]  n_unique={unique.size}")

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
        codes_b = codes_2d[mask]   # (n_cells_in_batch, n_levels)
        # Stash the batch label for the title-generator helpers.
        adata_b.uns["_plot_batch_label"] = str(b)

        # multi-level summary
        if not args.no_summary:
            out_summary = out_dir / f"code_index_batch{b}_summary.png"
            plot_batch_summary(
                adata_b, codes_b, level_labels, out_summary,
                point_size=args.point_size,
                max_categorical=args.max_categorical,
                max_codes_in_legend=args.max_codes_in_legend,
                xlim=xlim, ylim=ylim,
            )
            print(f"  -> wrote {out_summary}")

        # one-panel per (batch, level)
        if not args.no_per_level:
            for q, lbl in enumerate(level_labels):
                slug = lbl.replace(" ", "")
                out_path = out_dir / f"code_index_batch{b}_{slug}.png"
                plot_one_panel(
                    adata_b, codes_b[:, q], lbl, out_path,
                    point_size=args.point_size,
                    max_categorical=args.max_categorical,
                    max_codes_in_legend=args.max_codes_in_legend,
                    xlim=xlim, ylim=ylim,
                )
                print(f"  -> wrote {out_path}")

    print()
    print("=" * 78)
    print(f"Done. Wrote figures to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
