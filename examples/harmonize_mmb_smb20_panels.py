"""
Prepare the mmb0-1b_smb1-20b_1p dataset for the cross-platform retrieval
experiment (train on 20 STARmap samples, hold out the MERFISH sample at
inference and see which codes its cells map to). Also stamp every output
file with the `.uns` keys SQUINT's blob builder expects so blob positions
are deterministic (sorted by `adata_batch_id` ascending = batch number).

Input folder (default):
    /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/mmb0-1b_smb1-20b_1p/
        harmonised_merfish_mouse_brain_239-batch_batch82_shared_genes.h5ad
        harmonised_starmap_plus_mouse_cns_batch{1..20}.h5ad

What this script does:
  1. Reads the MERFISH file's gene panel as the canonical reference
     (the curated 431 genes denoted by `_shared_genes` in the filename).
  2. Computes the GLOBAL intersection across {MERFISH, all STARmap files}
     so every output file ends up with the same gene set in the same order.
     This is what InMemoryDatasetBlob.process() requires (`pandas.DataFrame.equals`
     on `.var` across batches — not just same set, same row order, same dtypes).
  3. Reindexes every input file to that shared panel, drops all `.var`
     columns (probe ids, QC fields, ensembl-only-on-one-platform metadata
     etc. that would still trigger panel-mismatch errors), and writes the
     harmonised file to a sibling output directory with `_shared_genes.h5ad`
     suffix to mirror the MERFISH file's naming convention.
  4. For each sample, renders a spatial scatter (uniform + colored by
     the standard set of label keys when present in obs) under
     <ARTIFACTS_DIR>/dataset_preparation/<dataset_name>/spatial_plots/.
     Plots live in the reproducibility-repo artifacts tree alongside
     every other run artifact (NOT next to the .h5ad outputs on /lustre),
     keyed by dataset name so multiple harmonisation runs of different
     dataset folders don't collide.

After this, point the dataset blob builder at the output folder. The
existing `make_dataset_blob_config()` in run_squint_mmb_smb.py expects
all 21 files to share a panel — this script's output is that input.

Usage:
    python examples/harmonize_mmb_smb20_panels.py
    python examples/harmonize_mmb_smb20_panels.py --no-plot
    python examples/harmonize_mmb_smb20_panels.py --output-dir /custom/path

The script never modifies the input directory.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_INPUT_DIR = Path(
    "/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/mmb0-1b_smb1-20b_1p"
)
DEFAULT_REFERENCE_FILE = (
    "harmonised_merfish_mouse_brain_239-batch_batch82_shared_genes.h5ad"
)
DEFAULT_COLOR_KEYS = [
    "cell_type",
    "niche",
    "Sub_molecular_tissue_region",
    "ccf_region_name",
]

# Plots live in the squint-reproducibility artifacts tree (NOT next to the
# .h5ad outputs in /lustre), keyed by dataset name. This matches the
# convention used by every other artifact in this codebase
# (run dirs, predicted_adata, code_index_plots, umap_plots, metrics, ...
# all live under ARTIFACTS_DIR).
ARTIFACTS_DIR = Path("/nfs/team361/sb75/squint-reproducibility/artifacts")
DATASET_PREPARATION_DIR = ARTIFACTS_DIR / "dataset_preparation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_var_names(path: Path) -> List[str]:
    """Read just the var index from an h5ad file (backed mode for speed)."""
    a = ad.read_h5ad(path, backed="r")
    names = list(a.var_names)
    if a.isbacked:
        a.file.close()
    return names


def _strip_var_metadata(a: ad.AnnData) -> ad.AnnData:
    """Drop all `.var` columns + varm/varp so panel-equality across batches holds."""
    a.var = pd.DataFrame(index=a.var_names)
    for k in list(a.varm.keys()):
        del a.varm[k]
    for k in list(a.varp.keys()):
        del a.varp[k]
    return a


def _stamp_uns_and_cell_id(
        a: ad.AnnData,
        filename: str,
    ) -> Optional[int]:
    """
    Set `.uns['batch']` / `.uns['dataset_id']` / `.uns['tissue']` /
    `.uns['species']`, and fabricate `obs['cell_id']` if missing.

    These are the keys SQUINT's `InMemoryDatasetBlob.process()` reads:
      - `uns['batch']` of the form 'batchN' is parsed (`int(uns['batch'][5:])`)
        for `adata_batch_id`. After pass 2 the blob is sorted by
        `adata_batch_id` ascending — if every input has a parseable
        `uns['batch']`, blob positions match the batch number ordering
        deterministically. Without it, the loader falls back to file-
        alphabetical position which is order-by-string and nearly always
        not what you want.
      - `cell_id` containing 'batchN' is the historical fallback for
        per-cell batch one-hots; we no longer use it (densification now
        comes from `obs['batch']`), but we still populate it for
        backward-compat with older code paths.

    Returns the parsed batch number (int) or None if the filename doesn't
    contain a 'batchN' substring.
    """
    m = re.search(r"batch(\d+)", filename)
    if not m:
        print(f"  [{filename}] no 'batchN' in filename; uns/cell_id NOT stamped")
        return None
    batch_n = int(m.group(1))
    batch_str = f"batch{batch_n}"

    # MERFISH vs STARmap: pick a stable dataset_id for downstream FiLM /
    # decoder-covariate code. Convention matches the existing
    # patch_anndata_uns() in run_squint_mmb_smb.py: 'mmb0' for MERFISH,
    # 'smb1' for STARmap.
    if "merfish" in filename.lower():
        dataset_id = "mmb0"
    elif "starmap" in filename.lower():
        dataset_id = "smb1"
    else:
        dataset_id = "unknown"

    a.uns["batch"]      = batch_str
    a.uns["dataset_id"] = dataset_id
    # Tissue/species defaults match what's already on the existing
    # MERFISH/STARmap files in `patch_anndata_uns`.
    a.uns.setdefault("tissue",  "mouse_brain")
    a.uns.setdefault("species", "mouse")

    if "cell_id" not in a.obs.columns:
        a.obs["cell_id"] = [
            f"{dataset_id}_{batch_str}_{i}" for i in range(a.n_obs)
        ]

    # Also stamp obs['batch'] when missing — this is the canonical column
    # `build_batch_one_hot_from_obs` densifies for batch-correction; on
    # files where it wasn't populated upstream we use the parsed batch
    # number string.
    if "batch" not in a.obs.columns:
        a.obs["batch"] = batch_str

    return batch_n


def _plot_one_sample(
        adata: ad.AnnData,
        stem: str,
        color_keys: List[str],
        out_dir: Path,
        point_size: float,
    ) -> None:
    """
    Per-sample spatial scatter. Always saves an uncoloured panel (just
    tissue shape). When color_keys are present in obs, also saves one
    coloured panel per key. PNG at 150 dpi — fine for sanity checks
    and small enough to commit / share.
    """
    if "spatial" not in adata.obsm:
        print(f"  [{stem}] obsm['spatial'] missing; skipping plot")
        return

    xy = np.asarray(adata.obsm["spatial"])

    # ---- uniform-grey layout (always) -------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c="lightgray",
               alpha=0.6, rasterized=True, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"{stem} (n={adata.n_obs:,})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_spatial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot: {stem}_spatial.png")

    # ---- one panel per available color key --------------------------------
    available = [c for c in color_keys if c in adata.obs.columns]
    for ck in available:
        try:
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=ck,
                size=point_size,
                show=False,
                title=f"{stem} — {ck}",
            )
            ax = plt.gca()
            ax.set_aspect("equal")
            for coll in ax.collections:
                coll.set_rasterized(True)
            plt.gcf().savefig(
                out_dir / f"{stem}_spatial_by_{ck}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            print(f"  plot: {stem}_spatial_by_{ck}.png")
        except Exception as exc:
            print(f"  [{stem}] plot by {ck} failed: {exc}")
            plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Folder containing the original .h5ad files. "
             f"Default: {DEFAULT_INPUT_DIR}.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Destination for harmonised .h5ad files. "
             "Default: <input-dir>_shared_genes/ (sibling of input).",
    )
    ap.add_argument(
        "--plot-dir", type=Path, default=None,
        help=f"Destination for the per-sample spatial plots. "
             f"Default: {DATASET_PREPARATION_DIR}/<dataset_name>/spatial_plots/ "
             f"where <dataset_name> = the OUTPUT directory's basename. "
             f"Plots live under the reproducibility-repo artifacts tree "
             f"so multiple harmonisation runs of different dataset folders "
             f"don't collide.",
    )
    ap.add_argument(
        "--reference-file", type=str, default=DEFAULT_REFERENCE_FILE,
        help="Filename within --input-dir whose gene panel is the "
             "canonical reference (everyone else is reindexed to its "
             "intersection with this panel, in this panel's order). "
             f"Default: {DEFAULT_REFERENCE_FILE}.",
    )
    ap.add_argument(
        "--color-keys", type=str, default=",".join(DEFAULT_COLOR_KEYS),
        help="Comma-separated obs columns to color spatial plots by. "
             "Missing columns are skipped silently.",
    )
    ap.add_argument(
        "--no-plot", action="store_true",
        help="Skip the per-sample spatial plotting step.",
    )
    ap.add_argument(
        "--point-size", type=float, default=2.0,
        help="Scatter point size for spatial plots (default 2.0).",
    )
    args = ap.parse_args()

    in_dir = args.input_dir
    if not in_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist or is not a directory: {in_dir}")

    out_dir = args.output_dir or in_dir.parent / f"{in_dir.name}_shared_genes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plots default to the artifacts tree (squint-reproducibility), keyed
    # by the OUTPUT folder's name (the "dataset name" downstream code will
    # use). This co-locates them with every other run artifact and avoids
    # writing plot blobs into the /lustre data folder.
    if args.plot_dir is not None:
        plot_dir = args.plot_dir
    else:
        plot_dir = DATASET_PREPARATION_DIR / out_dir.name / "spatial_plots"
    if not args.no_plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    ref_path = in_dir / args.reference_file
    if not ref_path.exists():
        raise SystemExit(f"reference file not found: {ref_path}")

    color_keys = [k.strip() for k in args.color_keys.split(",") if k.strip()]

    # ----- Pass 1: collect var_names and compute the shared panel ---------
    print(f"Input  : {in_dir}")
    print(f"Output : {out_dir}")
    print(f"Reference panel: {ref_path.name}")

    other_files = sorted(p for p in in_dir.glob("*.h5ad") if p != ref_path)
    print(f"Found {len(other_files)} non-reference .h5ad files.")

    print("\n=== Pass 1: scanning gene panels ===")
    ref_names = _read_var_names(ref_path)
    ref_set = set(ref_names)
    print(f"  {ref_path.name}: {len(ref_names)} genes (reference)")

    panels = [ref_set]
    for p in other_files:
        names = _read_var_names(p)
        panels.append(set(names))
        print(f"  {p.name}: {len(names)} genes")

    shared = set.intersection(*panels)
    if not shared:
        raise SystemExit("Empty intersection across all files — nothing to harmonise.")

    # Preserve REFERENCE order so MERFISH-style downstream tools see the
    # same column order they expect.
    final_order = [g for g in ref_names if g in shared]
    print(f"\nShared panel: {len(final_order)} genes")
    dropped = len(ref_names) - len(final_order)
    if dropped:
        print(f"  dropped {dropped} reference genes that are missing from at "
              f"least one STARmap file")

    # ----- Pass 2: reindex + write each file ------------------------------
    print("\n=== Pass 2: writing harmonised files ===")

    # Reference (MERFISH): keep the same filename in the output dir.
    ref = ad.read_h5ad(ref_path)
    n_var_in = ref.n_vars
    ref = ref[:, final_order].copy()
    ref = _strip_var_metadata(ref)
    ref_batch_n = _stamp_uns_and_cell_id(ref, ref_path.name)
    ref_out_path = out_dir / ref_path.name
    ref.write_h5ad(ref_out_path)
    print(f"  -> {ref_out_path.name}  "
          f"n_obs={ref.n_obs:,}  n_vars {n_var_in} -> {ref.n_vars}  "
          f"uns['batch']={ref.uns.get('batch')!r}  "
          f"dataset_id={ref.uns.get('dataset_id')!r}")
    if not args.no_plot:
        _plot_one_sample(ref, ref_out_path.stem, color_keys,
                         plot_dir, args.point_size)
    del ref

    # STARmap files: append `_shared_genes` to the stem if not already there.
    for p in other_files:
        a = ad.read_h5ad(p)
        n_var_in = a.n_vars
        a = a[:, final_order].copy()
        a = _strip_var_metadata(a)
        # Stamp uns/cell_id BEFORE renaming — uses the input filename
        # (which carries 'batchN') to derive the batch number.
        _stamp_uns_and_cell_id(a, p.name)

        stem = p.stem
        if "_shared_genes" not in stem:
            out_name = f"{stem}_shared_genes.h5ad"
        else:
            out_name = p.name
        out_path = out_dir / out_name
        a.write_h5ad(out_path)
        print(f"  -> {out_name}  "
              f"n_obs={a.n_obs:,}  n_vars {n_var_in} -> {a.n_vars}  "
              f"uns['batch']={a.uns.get('batch')!r}  "
              f"dataset_id={a.uns.get('dataset_id')!r}")

        if not args.no_plot:
            _plot_one_sample(a, Path(out_name).stem, color_keys,
                             plot_dir, args.point_size)
        del a

    print(f"\nDone.")
    print(f"  Harmonised files: {out_dir}")
    if not args.no_plot:
        print(f"  Spatial plots:    {plot_dir}")
    print(f"\nNext steps:")
    print(f"  1. Add make_dataset_blob_config_<name>() in run_squint_mmb_smb.py")
    print(f"     pointing data_directory_path at {out_dir.parent} and dataset")
    print(f"     name at {out_dir.name!r}.")
    print(f"  2. Build the blob: python examples/run_squint_mmb_smb.py "
          f"--build-blob --build-blob-dataset <name>")
    print(f"  3. Add a holdout-aware ablation patch (e.g. train on STARmap "
          f"batches 1..20 with adata_batch_idx=[1..20], hold out the "
          f"MERFISH file via SpatialBatchSplit val_batches).")


if __name__ == "__main__":
    main()
