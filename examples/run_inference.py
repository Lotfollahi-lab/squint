"""
Re-run inference + analysis (steps 2-6 of run_squint.py's full pipeline)
against a checkpoint that was already trained.

The script takes a `--variant` and `--timestamp`, looks up the run_dir
under `<ARTIFACTS_DIR>/<dataset>/<variant>/<timestamp>/` (resolving the
dataset name from the variant's build config), and re-runs:

   2. predict()                  -> predicted_adata.h5ad  (overwrites if
                                    one already exists in the run_dir)
   3. plot_code_indices_spatial  -> code_index_plots/
   4. plot_svg_reconstruction    -> svg_plots/
   5. plot_latent_umap           -> umap_plots/
   6. compute_inference_metrics  -> metrics/

All outputs land in the run_dir alongside the existing checkpoints + the
saved training config — same layout as `run_squint.py --all` produces.

Usage
-----
    python examples/run_inference.py \\
        --variant smoke-test+mmb0-1b_smb1-1b_1p \\
        --timestamp 20260507_081017

    # Cell-type labels for UMAPs / metrics:
    python examples/run_inference.py \\
        --variant dualvq+wide+rvq-both+decoder-cov+adv+chl59-8b_1p \\
        --timestamp 20260507_120000 \\
        --label-keys cell_type,niche

    # Override silver dir if the AnnDatas moved since training:
    python examples/run_inference.py \\
        --variant dualvq+narrow+rvq-both+decoder-cov+adv+mmb0-1b_smb1-20b_1p \\
        --timestamp 20260507_120000 \\
        --silver-dir /alt/path/to/silver
"""

# Silence the same upstream FutureWarnings run_squint.py filters at startup.
import warnings

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
try:
    import dask
    dask.config.set({"dataframe.query-planning": True})
except Exception:  # noqa: BLE001
    pass

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from run_squint import (  # noqa: E402
    VARIANTS,
    resolve_run_dir,
    run_inference_and_analysis,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Re-run inference + analysis on an already-trained checkpoint. "
            "Looks up the run_dir from --variant + --timestamp."
        ),
    )
    p.add_argument(
        "--variant", type=str, required=True,
        help="Registered variant name (must match what was used at train "
             "time). Use `python examples/run_squint.py --list-variants` "
             "to see the catalogue.",
    )
    p.add_argument(
        "--timestamp", type=str, required=True,
        help="Run timestamp (the YYYYMMDD_HHMMSS folder name under "
             "<ARTIFACTS_DIR>/<dataset>/<variant>/).",
    )
    p.add_argument("--list-variants", action="store_true",
                   help="Print all registered variants and exit.")
    p.add_argument(
        "--silver-dir", type=str, default=None,
        help="Override the folder of .h5ad files to run inference on. "
             "Defaults to the silver folder recorded in the training "
             "config (root_data_dir/silver/<dataset_name>).",
    )
    p.add_argument(
        "--model-ckpt-fname", type=str, default=None,
        help="Optional explicit checkpoint .ckpt path. Defaults to the "
             "best checkpoint under <run_dir>/checkpoints/.",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Override where predicted_adata.h5ad + plots + metrics land. "
             "Defaults to the run_dir itself.",
    )
    p.add_argument(
        "--label-keys", type=str,
        default="cell_type,niche,Sub_molecular_tissue_region,ccf_region_name",
        help="Comma-separated obs columns to colour UMAPs by. Default "
             "covers both the mouse-brain (cell_type, "
             "Sub_molecular_tissue_region, ccf_region_name) and CosMx Lung "
             "(cell_type, niche) niche label conventions; "
             "missing columns are silently skipped per UMAP.",
    )
    p.add_argument(
        "--batch-rename", type=str, default=None,
        help="Optional comma-separated rename for adata_batch_id "
             "categories in UMAPs (e.g. 'MERFISH,STARmap PLUS').",
    )
    p.add_argument(
        "--cell-label-keys", type=str, default=None,
        help="Override cell-code label set for compute_inference_metrics. "
             "Default: cell_type,cell_types.",
    )
    p.add_argument(
        "--niche-label-keys", type=str, default=None,
        help="Override niche-code label set for compute_inference_metrics. "
             "Default: niche,Sub_molecular_tissue_region,ccf_region_name.",
    )
    # Step-skipping flags. Each maps onto the matching parameter of
    # `run_inference_and_analysis()`. Useful when re-running the pipeline
    # against an existing predicted_adata.h5ad (e.g. to refresh metrics
    # after a metrics-only fix) or when one of the heavy steps (UMAP) was
    # already done satisfactorily and you only need the cheap ones.
    p.add_argument("--skip-predict", action="store_true",
                   help="Reuse the existing predicted_adata.h5ad in run_dir / "
                        "output_dir (errors if missing). Skips step 2.")
    p.add_argument("--skip-code-index-plots", action="store_true",
                   help="Skip plot_code_indices_spatial (step 3).")
    p.add_argument("--skip-svg-plots", action="store_true",
                   help="Skip plot_svg_reconstruction (step 4).")
    p.add_argument("--skip-umap", action="store_true",
                   help="Skip plot_latent_umap (step 5). The slowest step; "
                        "skip when prior UMAPs are still valid.")
    p.add_argument("--skip-metrics", action="store_true",
                   help="Skip compute_inference_metrics (step 6).")
    args = p.parse_args()

    if args.list_variants:
        print("Registered variants:")
        for name in VARIANTS:
            print(f"  {name}")
        return

    run_dir = resolve_run_dir(args.variant, args.timestamp)
    print(f"Resolved run_dir: {run_dir}")

    run_inference_and_analysis(
        run_dir=str(run_dir),
        silver_dir=args.silver_dir,
        output_dir=args.output_dir,
        model_ckpt_fname=args.model_ckpt_fname,
        label_keys=args.label_keys,
        batch_rename=args.batch_rename,
        cell_label_keys=args.cell_label_keys,
        niche_label_keys=args.niche_label_keys,
        skip_predict=args.skip_predict,
        skip_code_index_plots=args.skip_code_index_plots,
        skip_svg_plots=args.skip_svg_plots,
        skip_umap=args.skip_umap,
        skip_metrics=args.skip_metrics,
    )


if __name__ == "__main__":
    main()
