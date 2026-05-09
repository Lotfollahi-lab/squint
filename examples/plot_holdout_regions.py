"""
Visualise the per-batch held-out regions used by the
`+region-holdout+mmb0-1b_smb1-1b_1p` downstream task.

For each silver AnnData, plot the cells via `sc.pl.spatial` (with
`spot_size=auto-from-coords`) and colour by `data_split`:
  - "train" cells: muted grey
  - "test"  cells: prominent red — the held-out patch

The script:
  1. Reads `test_regions` from a saved training config
     (`<run_dir>/user_specified_config.yaml`) OR from explicit JSON via
     --test-regions-json. The first form is the common case; pass it
     `--run-dir <path>` and the script picks up exactly what was
     trained on.
  2. Loads each silver AnnData (silver_dir is auto-derived from the
     config or overridable via --silver-dir).
  3. Resolves percentile bounds against THIS batch's xy range, mirroring
     `SpatialBatchSplit._is_in_single_region`.
  4. Stamps `obs["data_split"]` on each AnnData (in memory only; the
     silver files are NOT modified) and renders one PNG/SVG per
     section.

Output (per section) goes to `<out_dir>/<source_filename>.holdout.png`
(default `<run_dir>/holdout_plots/` if --run-dir was given, else
`./holdout_plots/`).

Usage:
  # Use the regions from a trained run:
  python examples/plot_holdout_regions.py \\
      --run-dir /nfs/.../artifacts/mmb0-1b_smb1-1b_1p/<variant>/<timestamp>

  # Or pass regions explicitly (e.g. before training):
  python examples/plot_holdout_regions.py \\
      --silver-dir /lustre/.../silver/mmb0-1b_smb1-1b_1p_coord_aligned \\
      --test-regions-json '{"15": {"x_min_pct": 0.10, "x_max_pct": 0.35,
        "y_min_pct": 0.55, "y_max_pct": 0.80},
        "82": {"x_min_pct": 0.55, "x_max_pct": 0.80,
        "y_min_pct": 0.20, "y_max_pct": 0.45}}'
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional

# Match the warning suppression in run_squint.py (avoid noise from
# transitive deps at import time).
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
import scanpy as sc
import yaml

# Make SVG text editable in Illustrator / Inkscape / browsers.
# Default `svg.fonttype='path'` converts every glyph to an outlined path,
# so labels can no longer be edited as text downstream. `'none'` keeps
# text elements as <text> nodes referring to the system font by name —
# the SVG then opens with editable text wherever the font is available
# (or any reasonable substitute). `pdf.fonttype=42` is the analogous
# setting for PDF saves (TrueType embed instead of Type-3 outlines).
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


def _save_dual(fig, out_path, **savefig_kwargs) -> None:
    """Save the figure as BOTH `.png` and `.svg` siblings.

    Per-extension overrides:
      - SVG saves never use `dpi` (vector format; the kwarg is ignored
        for the rasterised parts but matplotlib still warns occasionally).
      - PNG saves keep whatever was passed in.
    """
    out_path = Path(out_path)
    png_kwargs = dict(savefig_kwargs)
    svg_kwargs = {k: v for k, v in savefig_kwargs.items() if k != "dpi"}
    fig.savefig(out_path.with_suffix(".png"), **png_kwargs)
    fig.savefig(out_path.with_suffix(".svg"), **svg_kwargs)


# Platform detection: filename substring -> display name.
# Used to build per-panel titles + the combined figure suptitle. Keys are
# lowercased substrings; first match wins. Override per-call via the
# CLI `--platforms` flag.
_DEFAULT_PLATFORM_PATTERNS = [
    ("merfish",  "MERFISH"),
    ("starmap",  "STARmap"),
    ("cosmx",    "CosMx"),
    ("xenium",   "Xenium"),
    ("visium",   "Visium"),
]


def _detect_platform(filename: str, overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Return a display name for the platform inferred from `filename`,
    or None if no match. `overrides` lets the caller add / override
    patterns (lowercase substring -> display name)."""
    name_lc = filename.lower()
    if overrides:
        for substr, display in overrides.items():
            if substr.lower() in name_lc:
                return display
    for substr, display in _DEFAULT_PLATFORM_PATTERNS:
        if substr in name_lc:
            return display
    return None


def _resolve_region_bounds(region: dict, x_lo, x_hi, y_lo, y_hi) -> tuple:
    """Return (x_min, x_max, y_min, y_max) — absolute bounds, percentiles
    resolved against the section's coord range. Mirrors
    `SpatialBatchSplit._is_in_single_region`."""
    def _resolve(abs_key, pct_key, lo, hi, default):
        if abs_key in region and region[abs_key] is not None:
            return float(region[abs_key])
        if pct_key in region and region[pct_key] is not None:
            return lo + float(region[pct_key]) * (hi - lo)
        return default
    x_min = _resolve("x_min", "x_min_pct", x_lo, x_hi, x_lo)
    x_max = _resolve("x_max", "x_max_pct", x_lo, x_hi, x_hi)
    y_min = _resolve("y_min", "y_min_pct", y_lo, y_hi, y_lo)
    y_max = _resolve("y_max", "y_max_pct", y_lo, y_hi, y_hi)
    return x_min, x_max, y_min, y_max


def _mark_holdout(adata: ad.AnnData, region) -> int:
    """Mark cells inside `region` (single dict or list of dicts) with
    obs["data_split"] = "test"; everything else "train". Returns
    n_test."""
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    else:
        raise SystemExit(
            f"AnnData missing obsm['spatial']; can't visualise without "
            f"per-cell coordinates."
        )
    x_lo, y_lo = xy.min(axis=0)
    x_hi, y_hi = xy.max(axis=0)
    regions_list = region if isinstance(region, list) else [region]

    in_any = np.zeros(adata.n_obs, dtype=bool)
    for r in regions_list:
        x_min, x_max, y_min, y_max = _resolve_region_bounds(
            r, x_lo, x_hi, y_lo, y_hi,
        )
        in_box = (
            (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max)
            & (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
        )
        in_any |= in_box

    adata.obs["data_split"] = np.where(in_any, "test", "train")
    return int(in_any.sum())


def _derive_adata_batch_id(adata: ad.AnnData, fallback_path: Path) -> Optional[int]:
    """Match `InMemoryDatasetBlob._derive_adata_batch_id`'s parsing rules
    on the silver file. Returns None if uns['batch'] isn't set / not
    parseable (in which case the caller should warn + skip)."""
    val = adata.uns.get("batch", None)
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, str):
        if val.startswith("batch"):
            try:
                return int(val[5:])
            except ValueError:
                pass
        try:
            return int(val)
        except ValueError:
            pass
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default=None,
                   help="Trained run directory containing "
                        "user_specified_config.yaml. test_regions are read "
                        "from there.")
    p.add_argument("--silver-dir", type=str, default=None,
                   help="Folder containing the .h5ad files. Defaults to "
                        "the silver dir derived from the saved config "
                        "(<root_data_dir>/silver/<dataset_name>).")
    p.add_argument("--test-regions-json", type=str, default=None,
                   help="JSON literal mapping batch_id -> region_spec "
                        "(absolute or *_pct keys). Overrides whatever's "
                        "in the saved config.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write the holdout plots. Defaults to "
                        "<run_dir>/holdout_plots/ if --run-dir was given, "
                        "else ./holdout_plots/.")
    p.add_argument("--spot-size", type=float, default=None,
                   help="Forwarded to sc.pl.spatial. Default lets scanpy "
                        "auto-pick.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--combined-out", type=str, default="holdout_preview",
        help="Stem for the combined figure with one panel per section, "
             "written as <out_dir>/<stem>.{png,svg}. Set to '' to skip. "
             "Default: 'holdout_preview'.",
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
        help="Skip the per-section .holdout.{png,svg} files (only emit "
             "the combined figure).",
    )
    args = p.parse_args()

    # Parse --platforms into a dict.
    platform_overrides: Optional[Dict[str, str]] = None
    if args.platforms:
        platform_overrides = {}
        for token in args.platforms.split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            k, v = token.split("=", 1)
            platform_overrides[k.strip()] = v.strip()

    config: dict = {}
    if args.run_dir:
        cfg_path = Path(args.run_dir) / "user_specified_config.yaml"
        if not cfg_path.exists():
            raise SystemExit(f"No config at {cfg_path}")
        with open(cfg_path) as f:
            config = yaml.safe_load(f) or {}

    # Resolve test_regions.
    if args.test_regions_json:
        test_regions = json.loads(args.test_regions_json)
    else:
        test_regions = (
            config.get("dataset", {})
                  .get("train_transform_params", {})
                  .get("test_regions", None)
        )
    if not test_regions:
        raise SystemExit(
            "No test_regions found. Pass --test-regions-json or use a "
            "--run-dir whose saved config has test_regions set (e.g. the "
            "dualvq+...+region-holdout+... variant)."
        )
    # Normalise keys to int.
    test_regions = {int(k): v for k, v in test_regions.items()}
    print(f"test_regions: {test_regions}")

    # Resolve silver_dir.
    if args.silver_dir:
        silver_dir = Path(args.silver_dir)
    else:
        root = Path(config["dataset"]["root_data_dir"])
        name = config["dataset"]["dataset_name"]
        silver_dir = root / "silver" / name
    print(f"silver_dir: {silver_dir}")

    silver_files = sorted(silver_dir.glob("*.h5ad"))
    if not silver_files:
        raise SystemExit(f"No .h5ad files under {silver_dir}.")

    # Resolve out_dir.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.run_dir:
        out_dir = Path(args.run_dir) / "holdout_plots"
    else:
        out_dir = Path("./holdout_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    sc.settings.set_figure_params(dpi=args.dpi, frameon=False)

    # First pass: load + mark every section that has a region. Keep the
    # marked AnnDatas so we can render BOTH per-section files and a
    # combined figure without re-reading h5ad twice.
    rendered: list[dict] = []  # entries: {bid, filename, adata, platform, spot_size}
    for f in silver_files:
        a = ad.read_h5ad(f)
        bid = _derive_adata_batch_id(a, f)
        if bid is None:
            print(f"  skip {f.name}: no parseable uns['batch']")
            continue
        if bid not in test_regions:
            print(f"  skip {f.name}: adata_batch_id={bid} not in "
                  f"test_regions {sorted(test_regions)}")
            continue

        n_test = _mark_holdout(a, test_regions[bid])
        print(f"  batch {bid:>3d} | {f.name}: "
              f"n_obs={a.n_obs:>6d}  n_test={n_test:>6d}  "
              f"({100.0 * n_test / max(a.n_obs, 1):.1f}%)")

        # sc.pl.spatial needs `obsm["spatial"]` (already there) and a
        # color key with at least 2 categories.
        a.obs["data_split"] = a.obs["data_split"].astype(
            "category"
        ).cat.set_categories(["train", "test"])

        if args.spot_size is not None:
            spot = float(args.spot_size)
        else:
            # Auto-pick a spot size proportional to coord range so cells
            # don't render as 1 pixel each.
            xy = np.asarray(a.obsm["spatial"], dtype=float)[:, :2]
            extent = max(xy[:, 0].ptp(), xy[:, 1].ptp())
            spot = max(extent / 200.0, 1.0)

        platform = _detect_platform(f.name, platform_overrides)
        rendered.append({
            "bid": bid, "filename": f.name, "adata": a,
            "platform": platform, "spot_size": spot,
        })

    if not rendered:
        raise SystemExit(
            "No sections matched test_regions. Make sure the silver "
            "files have uns['batch'] set and that the keys match "
            f"{sorted(test_regions)}."
        )

    palette = {"train": "#cccccc", "test": "#d62728"}

    # ---- per-section files (one .png + .svg per section) ----------------
    if not args.no_per_section:
        for r in rendered:
            # Title precedence: detected platform > "batch <bid>".
            title = r["platform"] if r["platform"] else f"batch {r['bid']}"
            fig = sc.pl.spatial(
                r["adata"],
                color="data_split",
                palette=palette,
                title=title,
                spot_size=r["spot_size"],
                show=False,
                return_fig=True,
            )
            out_path = out_dir / f"{Path(r['filename']).stem}.holdout.png"
            _save_dual(fig, out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
        print(f"Wrote {len(rendered)} per-section plot(s) to {out_dir}")

    # ---- combined figure (one figure, one panel per section) ------------
    if args.combined_out:
        # Suptitle: explicit override > " and ".join of detected platforms.
        if args.combined_title:
            suptitle = args.combined_title
        else:
            platforms_seen = [r["platform"] for r in rendered if r["platform"]]
            suptitle = (" and ".join(platforms_seen) if platforms_seen
                        else "Held-out regions")

        n = len(rendered)
        fig, axes = plt.subplots(
            1, n,
            figsize=(5.5 * n, 5.5),
            squeeze=False,
        )
        axes = axes.ravel()
        for ax, r in zip(axes, rendered):
            xy = np.asarray(r["adata"].obsm["spatial"], dtype=float)[:, :2]
            split = r["adata"].obs["data_split"].astype(str).to_numpy()
            colors = np.array([palette[s] for s in split])
            # Direct scatter so we control marker size and the SVG
            # contains a single Path collection (small file, all text
            # nodes intact). sc.pl.spatial wraps matplotlib but injects
            # a lot of ax-level decorations we don't need here.
            ax.scatter(
                xy[:, 0], xy[:, 1],
                c=colors, s=max(r["spot_size"] / 4.0, 0.4),
                linewidths=0, marker="o",
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()  # match scanpy/scanpy.spatial convention
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            panel_title = r["platform"] if r["platform"] else f"batch {r['bid']}"
            ax.set_title(panel_title, fontsize=14)

        # Single combined legend, top-right of the figure (avoids
        # cluttering each panel; the colour mapping is shared).
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['train'],
                   markersize=8, label='train'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['test'],
                   markersize=8, label='test (held out)'),
        ]
        fig.legend(
            handles=legend_handles, loc="upper right",
            bbox_to_anchor=(0.99, 0.97), frameon=False, fontsize=11,
        )
        fig.suptitle(suptitle, fontsize=16, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        combined_path = out_dir / args.combined_out
        _save_dual(fig, combined_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote combined plot to {combined_path}.{{png,svg}}")


if __name__ == "__main__":
    main()
