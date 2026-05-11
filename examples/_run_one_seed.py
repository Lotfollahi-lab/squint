#!/usr/bin/env python3
"""
Single-seed runner for the multi-seed SQUINT submission. Invoked by
`_run_one_seed.sh` inside one LSF job.

Responsibilities:
  1. Monkey-patch the variant's `build()` so `cfg["experiment"]["seed"]`
     is set to the seed we got from argv.
  2. Call `train(variant)` -> writes a fresh
     <ARTIFACTS_DIR>/<dataset_tag>/<variant>/<TS_seed_N>/ run dir.
  3. Call `run_inference_and_analysis(run_dir, ...)` -> writes
     `<run_dir>/metrics/*.csv` (+ optional plots).
  4. Stamp the resulting run_dir + wall-clock runtime into the sweep
     output dir so the aggregator (a separate LSF job, possibly running
     long after this seed finished) can find it::

         <OUT_DIR>/seed_runs/seed_<N>_run_dir.txt
         <OUT_DIR>/seed_runs/seed_<N>_runtime_seconds.txt
         <OUT_DIR>/seed_runs/seed_<N>_status.txt    ("OK" or error msg)

Usage (always invoked by `_run_one_seed.sh`, but works standalone too):

    python _run_one_seed.py <VARIANT> <SEED> <OUT_DIR> [--skip-umap ...]

Any extra args after OUT_DIR are forwarded to
`run_inference_and_analysis()` via the same flag names as
`run_inference.py` (`--skip-predict`, `--skip-code-index-plots`,
`--skip-svg-plots`, `--skip-umap`, `--skip-metrics`,
`--silver-dir`, `--label-keys`).
"""
from __future__ import annotations

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
import time
import traceback
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# pylint: disable=wrong-import-position
from run_squint import (  # noqa: E402
    ARTIFACTS_DIR,
    DATASET_NAME,
    VARIANTS,
    train,
    run_inference_and_analysis,
)


def _patch_seed(variant: str, seed: int):
    """Replace `VARIANTS[variant]["build"]` with a seeded version.
    Returns the original build so the caller can restore it.

    `train()` calls `VARIANTS[variant]["build"]()` itself, so we
    override the registry rather than passing the seed through any
    other channel."""
    original = VARIANTS[variant]["build"]

    def _seeded(_orig=original, _seed=int(seed)):
        cfg = _orig()
        cfg["experiment"]["seed"] = int(_seed)
        return cfg

    VARIANTS[variant]["build"] = _seeded
    return original


def _resolve_dataset_tag(variant: str) -> str:
    cfg = VARIANTS[variant]["build"]()
    return cfg["dataset"].get(
        "dataset_tag",
        cfg["dataset"].get("dataset_name", DATASET_NAME),
    )


def _write_stamp(seed_runs_dir: Path, seed: int, name: str, value: str) -> None:
    """Write a single-line stamp file `seed_<N>_<name>.txt` in
    `<OUT_DIR>/seed_runs/`. Used by the aggregator to discover which
    SQUINT run_dir each seed produced (+ runtime / status)."""
    seed_runs_dir.mkdir(parents=True, exist_ok=True)
    out = seed_runs_dir / f"seed_{seed}_{name}.txt"
    out.write_text(str(value).rstrip("\n") + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant", type=str)
    p.add_argument("seed", type=int)
    p.add_argument("out_dir", type=Path)
    # Forwarded to run_inference_and_analysis. Defaults match the
    # benchmark-figure-only workflow (metrics yes, plots no — UMAPs are
    # the slowest step and not needed for per_seed_*.csv aggregates).
    p.add_argument("--silver-dir", type=str, default=None)
    p.add_argument("--label-keys", type=str,
                   default="cell_type,niche,Sub_molecular_tissue_region,ccf_region_name")
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--skip-code-index-plots", action="store_true")
    p.add_argument("--skip-svg-plots", action="store_true")
    p.add_argument("--skip-umap", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    args = p.parse_args()

    if args.variant not in VARIANTS:
        raise SystemExit(
            f"Unknown variant {args.variant!r}. Use "
            f"`python examples/run_squint.py --list-variants` to list."
        )

    seed_runs_dir = args.out_dir / "seed_runs"
    seed_runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Multi-seed runner: variant={args.variant} seed={args.seed}")
    print(f"  sweep out_dir : {args.out_dir}")
    print(f"  stamps dir    : {seed_runs_dir}")
    print("=" * 78)

    t0 = time.time()
    original_build = _patch_seed(args.variant, args.seed)
    try:
        try:
            run_dir = train(args.variant)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\nTRAIN FAILED on seed {args.seed}: {type(e).__name__}: {e}")
            print(tb)
            _write_stamp(seed_runs_dir, args.seed, "status",
                         f"train_failed: {type(e).__name__}: {e}")
            return 2

        # Older train() signatures didn't return run_dir; resolve from
        # the variant + latest timestamp as a fallback.
        if run_dir is None:
            dataset_tag = _resolve_dataset_tag(args.variant)
            variant_slug = args.variant.replace("/", "_").replace(" ", "_")
            candidates = sorted(
                p for p in (ARTIFACTS_DIR / dataset_tag / variant_slug).glob("*/")
                if p.is_dir()
            )
            if not candidates:
                msg = (f"Could not auto-locate run_dir for "
                       f"{args.variant!r} after train()")
                _write_stamp(seed_runs_dir, args.seed, "status",
                             f"resolve_failed: {msg}")
                raise RuntimeError(msg)
            run_dir = str(candidates[-1])

        run_dir = str(run_dir)
        print(f"\n[seed {args.seed}] train complete, run_dir={run_dir}")
        # Stamp the run_dir EARLY (before inference) so the aggregator
        # can still find the training artifacts even if inference fails.
        _write_stamp(seed_runs_dir, args.seed, "run_dir", run_dir)

        try:
            run_inference_and_analysis(
                run_dir=run_dir,
                silver_dir=args.silver_dir,
                label_keys=args.label_keys,
                skip_predict=args.skip_predict,
                skip_code_index_plots=args.skip_code_index_plots,
                skip_svg_plots=args.skip_svg_plots,
                skip_umap=args.skip_umap,
                skip_metrics=args.skip_metrics,
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\nINFERENCE FAILED on seed {args.seed}: "
                  f"{type(e).__name__}: {e}")
            print(tb)
            _write_stamp(seed_runs_dir, args.seed, "status",
                         f"inference_failed: {type(e).__name__}: {e}")
            return 3

    finally:
        VARIANTS[args.variant]["build"] = original_build

    seconds = time.time() - t0
    _write_stamp(seed_runs_dir, args.seed, "runtime_seconds", f"{seconds:.3f}")
    _write_stamp(seed_runs_dir, args.seed, "status", "OK")
    print(f"\n[seed {args.seed}] done in {seconds:.1f}s -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
