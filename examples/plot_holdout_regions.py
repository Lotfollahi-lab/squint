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
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import yaml


def _save_dual(fig, out_path, **savefig_kwargs) -> None:
    """Save the figure as BOTH `.png` and `.svg` siblings."""
    out_path = Path(out_path)
    for ext in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(ext), **savefig_kwargs)


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
    args = p.parse_args()

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

    n_plotted = 0
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

        title = f"batch {bid} — {f.name}\nheld-out cells in red"
        plot_kwargs = dict(
            color="data_split",
            palette={"train": "#cccccc", "test": "#d62728"},
            title=title,
            show=False,
            return_fig=True,
        )
        if args.spot_size is not None:
            plot_kwargs["spot_size"] = float(args.spot_size)
        else:
            # Auto-pick a spot size proportional to coord range so cells
            # don't render as 1 pixel each.
            xy = np.asarray(a.obsm["spatial"], dtype=float)[:, :2]
            extent = max(xy[:, 0].ptp(), xy[:, 1].ptp())
            plot_kwargs["spot_size"] = max(extent / 200.0, 1.0)

        fig = sc.pl.spatial(a, **plot_kwargs)
        out_path = out_dir / f"{f.stem}.holdout.png"
        _save_dual(fig, out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        n_plotted += 1

    if n_plotted == 0:
        raise SystemExit(
            "No sections matched test_regions. Make sure the silver "
            "files have uns['batch'] set and that the keys match "
            f"{sorted(test_regions)}."
        )
    print(f"Wrote {n_plotted} plot(s) to {out_dir}")


if __name__ == "__main__":
    main()
