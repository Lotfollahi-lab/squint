"""
Plot the mmb0-1b_smb1-1b_1p mouse sections (1 MERFISH + 1 STARmap)
as plain spatial scatters — NO labels, every cell rendered in light
grey. Useful as a "tissue overview" figure for the paper / talks.

For each silver `.h5ad` under `--silver-dir`, render a single panel
of all cells coloured grey, titled by detected platform (MERFISH /
STARmap, auto-detected from filename — override via `--platforms`).
Also writes a combined `mmb_sections.{png,svg}` with both sections
side-by-side, suptitled `"MERFISH and STARmap"`.

Outputs are hybrid SVGs: the dense scatter is rasterised at the
requested DPI (so file sizes stay small + Illustrator/Inkscape don't
choke on 100k-cell PathCollections), while the title remains live
`<text>` (editable downstream). Same `svg.fonttype='none'` convention
as the other plotting scripts in this directory.

Usage:
    python examples/plot_mmb_sections.py
    # custom silver / output paths:
    python examples/plot_mmb_sections.py \\
        --silver-dir /nfs/.../silver/mmb0-1b_smb1-1b_1p \\
        --out-dir /nfs/.../dataset_preparation/mmb0-1b_smb1-1b_1p
    # only the combined figure:
    python examples/plot_mmb_sections.py --no-per-section
"""

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# Match the warning-suppression in run_squint.py + sibling plot scripts.
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Editable text in SVG (no glyph outlining). Matches plot_holdout_regions.py
# and plot_chl59_ground_truth.py.
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


# ---------------------------------------------------------------------------
# Helpers (mirrored from sibling plot scripts)
# ---------------------------------------------------------------------------

# Filename substring -> display name. Same map used in plot_holdout_regions.py;
# this file plots a different "view" (tissue overview, no labels) but should
# stay consistent with how the other scripts label the same sections.
_DEFAULT_PLATFORM_PATTERNS = [
    ("merfish",  "MERFISH"),
    ("starmap",  "STARmap"),
    ("cosmx",    "CosMx"),
    ("xenium",   "Xenium"),
    ("visium",   "Visium"),
]


def _detect_platform(filename: str, overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Infer a platform display name from `filename`. `overrides` is a
    {lowercase-substring -> display-name} map that takes precedence."""
    name_lc = filename.lower()
    if overrides:
        for substr, display in overrides.items():
            if substr.lower() in name_lc:
                return display
    for substr, display in _DEFAULT_PLATFORM_PATTERNS:
        if substr in name_lc:
            return display
    return None


def _save_dual(fig, out_path: Path, **savefig_kwargs) -> None:
    """Save figure as both `.png` and `.svg`. Both saves see the same
    `dpi` (controls full-figure raster for PNG and just the embedded
    raster layer in SVG; vector text stays editable at any dpi)."""
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), **savefig_kwargs)
    fig.savefig(out_path.with_suffix(".svg"), **savefig_kwargs)


def _spot_size_for(n_cells: int) -> float:
    """Default matplotlib scatter `s` (= marker area in points^2) given
    the cell count. Same density buckets as plot_chl59_ground_truth.py
    so the per-cell visual density looks consistent across datasets.
    Override per-call with --spot-size when individual cells still
    overlap or are too small."""
    if n_cells <= 0:
        return 2.0
    if n_cells >= 50_000:
        return 0.5
    if n_cells >= 20_000:
        return 1.0
    if n_cells >= 5_000:
        return 2.0
    return 4.0


def _scatter_grey(ax, xy: np.ndarray, spot_size: float,
                  title: str, color: str = "#bdbdbd") -> None:
    """Plain grey scatter on `ax`, no legend, axes off. The scatter is
    `rasterized=True` so it embeds as a single PNG layer in the SVG;
    the title is real <text> and stays editable."""
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=color, s=max(spot_size, 0.4),
        linewidths=0, marker="o",
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match scanpy spatial convention (y-down)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=14)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--silver-dir", type=str,
        default="/nfs/team361/sb75/DATASETS/silver/mmb0-1b_smb1-1b_1p",
        help="Folder containing the mmb0-1b_smb1-1b_1p .h5ad files. "
             "Default: /nfs/team361/sb75/DATASETS/silver/mmb0-1b_smb1-1b_1p",
    )
    p.add_argument(
        "--out-dir", type=str,
        default="/nfs/team361/sb75/squint-reproducibility/artifacts/"
                "dataset_preparation/mmb0-1b_smb1-1b_1p",
        help="Where to write the plots. Default: "
             "<ARTIFACTS>/dataset_preparation/mmb0-1b_smb1-1b_1p",
    )
    p.add_argument(
        "--combined-out", type=str, default="mmb_sections",
        help="Stem for the combined figure with one panel per section, "
             "written as <out_dir>/<stem>.{png,svg}. Set to '' to skip. "
             "Default: 'mmb_sections'.",
    )
    p.add_argument(
        "--combined-title", type=str, default=None,
        help='Suptitle for the combined figure. Default: " and ".join '
             "of detected platform names (e.g. 'MERFISH and STARmap').",
    )
    p.add_argument(
        "--platforms", type=str, default=None,
        help="Optional override mapping <filename-substring>=<display>, "
             "comma-separated. Example: 'merfish=MERFISH,starmap=STARmap'. "
             "Built-in detection covers MERFISH/STARmap/CosMx/Xenium/Visium "
             "by default; this flag is only needed for unusual filenames.",
    )
    p.add_argument(
        "--no-per-section", action="store_true",
        help="Skip the per-section .png/.svg files; only emit combined.",
    )
    p.add_argument(
        "--spot-size", type=float, default=None,
        help="Forwarded to matplotlib scatter `s` (= marker area in "
             "points^2). Default auto-picks from cell density.",
    )
    p.add_argument(
        "--cell-color", type=str, default="#bdbdbd",
        help="Hex/named colour for the grey cells. Default '#bdbdbd' "
             "(matplotlib light grey). Pick something darker (e.g. "
             "'#9e9e9e') for print contrast; lighter for a watermark "
             "feel.",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution of the rasterised tissue scatter. Vector text "
             "(title, suptitle) is unaffected by this. Default 300.",
    )
    args = p.parse_args()

    # Parse --platforms into a {lowercase substring -> display} dict.
    platform_overrides: Optional[Dict[str, str]] = None
    if args.platforms:
        platform_overrides = {}
        for token in args.platforms.split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            k, v = token.split("=", 1)
            platform_overrides[k.strip()] = v.strip()

    silver_dir = Path(args.silver_dir)
    if not silver_dir.is_dir():
        raise SystemExit(f"silver_dir does not exist: {silver_dir}")
    silver_files = sorted(silver_dir.glob("*.h5ad"))
    if not silver_files:
        raise SystemExit(f"No .h5ad files under {silver_dir}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"silver_dir : {silver_dir}")
    print(f"out_dir    : {out_dir}")
    print(f"sections   : {len(silver_files)}")

    # First pass: load each AnnData + resolve title.
    rendered: List[Dict] = []
    for f in silver_files:
        a = ad.read_h5ad(f)
        if "spatial" not in a.obsm:
            print(f"  skip {f.name}: missing obsm['spatial']")
            continue
        platform = _detect_platform(f.name, platform_overrides)
        title = platform if platform else f.stem
        spot = (
            float(args.spot_size) if args.spot_size is not None
            else _spot_size_for(a.n_obs)
        )
        rendered.append({
            "filename": f.name, "stem": f.stem,
            "title": title, "adata": a, "spot_size": spot,
        })
        print(f"  {f.name:60s}  n_obs={a.n_obs:>7d}  title={title!r}")

    if not rendered:
        raise SystemExit("No plottable sections found.")

    # Per-section files.
    if not args.no_per_section:
        for r in rendered:
            xy = np.asarray(r["adata"].obsm["spatial"], dtype=float)[:, :2]
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            _scatter_grey(ax, xy, r["spot_size"], title=r["title"],
                          color=args.cell_color)
            fig.tight_layout()
            out_path = out_dir / r["stem"]
            _save_dual(fig, out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
        print(f"Wrote {len(rendered)} per-section plot(s) to {out_dir}")

    # Combined figure (one panel per section, side-by-side).
    if args.combined_out:
        # Suptitle: explicit override > " and ".join of platform titles.
        if args.combined_title:
            suptitle = args.combined_title
        else:
            platforms_seen = [
                r["title"] for r in rendered
                if r["title"] in {p[1] for p in _DEFAULT_PLATFORM_PATTERNS}
                or (platform_overrides and r["title"] in platform_overrides.values())
            ]
            suptitle = (" and ".join(platforms_seen) if platforms_seen
                        else "Tissue sections")

        n = len(rendered)
        fig, axes = plt.subplots(
            1, n,
            figsize=(6.5 * n, 6.5),
            squeeze=False,
        )
        axes = axes.ravel()
        for ax, r in zip(axes, rendered):
            xy = np.asarray(r["adata"].obsm["spatial"], dtype=float)[:, :2]
            _scatter_grey(ax, xy, r["spot_size"], title=r["title"],
                          color=args.cell_color)
        fig.suptitle(suptitle, fontsize=18, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = out_dir / args.combined_out
        _save_dual(fig, out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote combined figure to {out_path}.{{png,svg}}")


if __name__ == "__main__":
    main()
