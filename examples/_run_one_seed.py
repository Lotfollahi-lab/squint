#!/usr/bin/env python3
"""
Single-seed runner for the multi-seed SQUINT submission. Invoked by
`_run_one_seed.sh` inside one LSF job.

Two-phase design (motivated by apples-to-apples runtime comparison vs
baseline methods):

  Phase 1 (TIMED — recorded in seed_<N>_runtime_seconds.txt):
    1. Monkey-patch the variant's `build()` so `cfg["experiment"]["seed"]`
       is set to the seed we got from argv.
    2. `train(variant)` -> writes a fresh
       <ARTIFACTS_DIR>/<dataset_tag>/<variant>/<TS_seed_N>/ run dir.
    3. `run_inference_and_analysis(... skip_plots=True, skip_metrics=True)`
       -> runs ONLY `predict()`, which writes `cell_code_index` /
       `neighborhood_code_index` to `predicted_adata.h5ad`. Those
       codes ARE the model's clusters — no downstream Leiden needed.

  Phase 2 (UNTIMED — for benchmark CSVs):
    4. `run_inference_and_analysis(skip_predict=True, skip_metrics=False)`
       -> populates `<run_dir>/metrics/*.csv` so the aggregator has
       per-seed NMI / ARI / iLISI / MMD / Pearson numbers to fold into
       the long-format `per_seed_*.csv` files. Optional plot steps
       (UMAP / code-index spatial / SVG reconstruction) also run here
       IF the caller enabled them via flags.

Why split: the recorded `runtime_seconds` should be a fair "time to
obtain clusters" number, comparable across methods. Metric computation
is benchmark scaffolding (not method cost), and visualisation plots
are downstream of the clusters — neither should be charged to the
method's runtime.

Stamps written to <OUT_DIR>/seed_runs/:
    seed_<N>_run_dir.txt              SQUINT run dir produced this seed
    seed_<N>_runtime_seconds.txt      phase 1 wall-clock (train + predict)
    seed_<N>_status.txt               "OK" / error message
    seed_<N>_runtime_methodology.txt  one-line explainer of what runtime_seconds covers

Usage (always invoked by `_run_one_seed.sh`, but works standalone too):

    python _run_one_seed.py <VARIANT> <SEED> <OUT_DIR> [--skip-umap ...]

Any extra args after OUT_DIR are forwarded to
`run_inference_and_analysis()` via the same flag names as
`run_inference.py` (`--skip-predict`, `--skip-code-index-plots`,
`--skip-svg-plots`, `--skip-umap`, `--skip-metrics`,
`--silver-dir`, `--label-keys`). `--skip-*-plots` and `--skip-umap`
only affect phase 2 since phase 1 always skips them. `--skip-metrics`
turns off the metric-CSV write in phase 2 (don't pass this unless
you want to re-aggregate from existing CSVs only).
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


def _patch_seed(
        variant: str,
        seed: int,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ):
    """Replace `VARIANTS[variant]["build"]` with a seeded version that
    also applies optional `batch_size` / `lr` overrides. Returns the
    original build so the caller can restore it.

    Why monkey-patch the registry: `train()` calls
    `VARIANTS[variant]["build"]()` itself with no kwargs. We can't pass
    the seed (or any other override) through `train`'s signature
    without breaking every other caller, so we replace the registry
    entry for the lifetime of THIS seed's job and restore it in the
    caller's `finally:` block.

    The same trick is the cleanest way to layer `batch_size` / `lr`
    overrides on top of the variant's build chain: the variant's own
    patches (rvq sizes, adversarial config, decoder-cov, etc.) run
    first, then we layer the runtime overrides on the resulting
    config. That order matters — a variant's build could otherwise
    legitimately reset `batch_size` (e.g. dataset-specific patches
    like `_patch_dual_mmb20` force batch_size=64 for memory).
    """
    original = VARIANTS[variant]["build"]

    def _seeded(_orig=original,
                _seed=int(seed),
                _bs=batch_size,
                _lr=lr):
        cfg = _orig()
        cfg["experiment"]["seed"] = int(_seed)
        if _bs is not None:
            cfg["datamodule"]["loader_params"]["batch_size"] = int(_bs)
        if _lr is not None:
            cfg["model"]["optimizer_params"]["lr"] = float(_lr)
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

    # Optional run-time overrides for the variant's config. Both are
    # layered on top of `VARIANTS[variant]["build"]()` inside
    # `_patch_seed`, so the variant's own patches (rvq sizes,
    # adversarial config, ...) run first and these overrides win.
    #
    # `--batch-size` overrides `loader_params["batch_size"]` (= 256 in
    # the base config). `--lr` overrides
    # `optimizer_params["lr"]` (= 5e-4 in the base config).
    #
    # Pair them sensibly: when you bump batch_size by 16x, scale lr
    # too. Default lr was tuned at batch=256; for batch=4096:
    #   - sqrt-scaling (recommended start): lr = 5e-4 * sqrt(16)
    #                                          = 2e-3
    #   - linear-scaling (more aggressive):   lr = 5e-4 * 16
    #                                          = 8e-3 — usually
    #                                          destabilises adversarial
    #                                          training; use only with
    #                                          a longer warmup.
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override `loader_params.batch_size` for this seed. "
             "Pair with a rescaled --lr (sqrt-scaling: lr ≈ 5e-4 * "
             "sqrt(batch/256)). Default: inherit from the variant.",
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="Override `optimizer_params.lr` for this seed. The base "
             "config sets lr=5e-4 at batch=256; rescale with batch size "
             "if you're changing both. Default: inherit from the variant.",
    )
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

    # ---- Phase 1 (TIMED): train + predict only ----------------------------
    # The recorded `runtime_seconds` covers ONLY the steps required to
    # produce the model's clusters from scratch. For SQUINT that's
    # `train()` (model fit) + `predict()` (writes
    # `cell_code_index` / `neighborhood_code_index` to
    # `predicted_adata.h5ad`). The codes ARE the clusters — no
    # downstream Leiden / UMAP needed.
    #
    # The AnnData WRITE step inside predict() is timed separately by
    # `predict()` itself (sidecar `predict_write_seconds.txt`) and
    # SUBTRACTED from this wall-clock total below — disk I/O is
    # benchmark scaffolding, not method cost, and the baseline runners
    # don't include it in their per-seed timers either.
    #
    # Metric computation (compute_inference_metrics) and any optional
    # plot steps run in phase 2 below, OUTSIDE the timer, so the
    # reported runtime is apples-to-apples with the per-seed numbers
    # produced by the baseline runners. See `runtime_methodology.txt`
    # written at the end of this script for the canonical methodology
    # statement.
    t0 = time.time()
    original_build = _patch_seed(
        args.variant, args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Print the RESOLVED batch_size + lr that train() will actually
    # consume, with provenance tag (variant-default vs CLI override).
    # This calls the patched build once for a peek — cheap (dict
    # manipulation only, no model init or file I/O) and lets every
    # LSF job log unambiguously show what config was used for this
    # seed, without the reader having to know whether overrides were
    # passed.
    _peek_cfg = VARIANTS[args.variant]["build"]()
    _resolved_bs = _peek_cfg["datamodule"]["loader_params"]["batch_size"]
    _resolved_lr = _peek_cfg["model"]["optimizer_params"]["lr"]
    _bs_src = "CLI override" if args.batch_size is not None else "variant default"
    _lr_src = "CLI override" if args.lr is not None else "variant default"
    print(f"[seed {args.seed}] resolved training config:")
    print(f"  batch_size = {_resolved_bs}  ({_bs_src})")
    print(f"  lr         = {_resolved_lr}  ({_lr_src})")
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
        # Stamp the run_dir EARLY (before predict) so the aggregator
        # can still find the training artifacts even if predict fails.
        _write_stamp(seed_runs_dir, args.seed, "run_dir", run_dir)

        # Phase-1 inference: run ONLY predict() (writes predicted_adata.h5ad
        # containing the code-index obs columns). Force-skip every other
        # step — plots and metrics belong in the untimed phase 2.
        #
        # The three `predict_*` overrides are the multi-seed-only
        # predict-time optimisations (ablations / direct `--all` runs
        # default to the legacy values inside predict()):
        #
        #   precision="bf16-mixed"
        #     Enables Lightning autocast on the predict forward.
        #     1.5-2x faster encoder / decoder MLPs on modern GPUs
        #     (H100 / H200 / A100). bf16 keeps the fp32 exponent range
        #     so we avoid the fp16 overflow risk, but the codebook
        #     `argmin` over squared L2 distances can drift by a few
        #     ULPs — a small number of borderline cells may pick a
        #     different code vs fp32. Use `--predict-precision none`
        #     (or wire an opt-out flag) if you need bit-exact
        #     reproducibility against an fp32 baseline.
        #   strategy="auto"
        #     Drops the legacy `ddp_find_unused_parameters_true` for
        #     predict, letting Lightning auto-pick
        #     `SingleDeviceStrategy` on single-GPU LSF jobs (the
        #     common case for multi-seed). ~5-15% predict speedup,
        #     no output change.
        #   compression="lzf"
        #     ~3-5x faster AnnData writes than gzip on NFS — only
        #     affects wall-clock UX since the write is already
        #     excluded from `runtime_seconds`. File ends up ~1.5x
        #     bigger on disk; readers (anndata.read_h5ad) handle lzf
        #     transparently.
        try:
            run_inference_and_analysis(
                run_dir=run_dir,
                silver_dir=args.silver_dir,
                label_keys=args.label_keys,
                skip_predict=args.skip_predict,    # respect --skip-predict
                                                   # if user pre-ran predict
                skip_code_index_plots=True,        # untimed: phase 2
                skip_svg_plots=True,               # untimed: phase 2
                skip_umap=True,                    # untimed: phase 2
                skip_metrics=True,                 # untimed: phase 2
                predict_precision="bf16-mixed",    # multi-seed only
                predict_strategy="auto",           # multi-seed only
                predict_compression="lzf",         # multi-seed only
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\nINFERENCE (predict) FAILED on seed {args.seed}: "
                  f"{type(e).__name__}: {e}")
            print(tb)
            _write_stamp(seed_runs_dir, args.seed, "status",
                         f"inference_failed: {type(e).__name__}: {e}")
            return 3

    finally:
        VARIANTS[args.variant]["build"] = original_build

    phase1_wallclock = time.time() - t0

    # Subtract the predicted_adata.h5ad WRITE time so `runtime_seconds`
    # excludes disk I/O. `predict()` (in run_squint.py) stamps the
    # write duration to `<run_dir>/predict_write_seconds.txt` after
    # the `adata.write_h5ad` call returns. Excluding the write matches
    # the baseline runners' convention (`_record_seed_runtime` in
    # `run_pca_leiden.py` etc., which time only model + clustering and
    # never include a downstream write or plot step). Best-effort: if
    # the stamp is missing or unreadable we fall back to the raw
    # wall-clock (same as pre-change behaviour) and warn so the user
    # can spot it.
    predict_write_seconds = 0.0
    write_stamp_path = Path(run_dir) / "predict_write_seconds.txt"
    if write_stamp_path.is_file():
        try:
            predict_write_seconds = float(write_stamp_path.read_text().strip())
        except (ValueError, OSError) as e:  # noqa: BLE001
            print(f"WARNING: could not parse {write_stamp_path} ({e}); "
                  "AnnData write time will NOT be subtracted from "
                  f"runtime_seconds for seed {args.seed}.")
    else:
        print(f"WARNING: {write_stamp_path} not found; AnnData write time "
              f"will NOT be subtracted from runtime_seconds for seed "
              f"{args.seed}. Was predict() interrupted before the write?")

    seconds = phase1_wallclock - predict_write_seconds
    _write_stamp(seed_runs_dir, args.seed, "runtime_seconds", f"{seconds:.3f}")
    print(f"\n[seed {args.seed}] phase 1 (train + predict) done in "
          f"{phase1_wallclock:.1f}s wall-clock; AnnData write "
          f"{predict_write_seconds:.1f}s excluded → "
          f"runtime_seconds = {seconds:.1f}s -> {run_dir}")

    # ---- Phase 2 (UNTIMED): metrics + any user-requested plots -----------
    # Decoupled from the runtime stamp on purpose: metrics produce the
    # benchmark CSVs (NMI / ARI / iLISI / MMD / Pearson) that the
    # aggregator reads, but per the apples-to-apples runtime principle,
    # the time spent producing those CSVs shouldn't be charged to the
    # method. Plots are skipped by default (`SKIP_*=1` in
    # `submit_multi_seed.sh`); if the caller re-enabled them via flags
    # they also run here, untimed. If everything is skipped, this is a
    # no-op (the function returns early without re-reading the predicted
    # adata).
    phase2_needed = not (
        args.skip_metrics
        and args.skip_code_index_plots
        and args.skip_svg_plots
        and args.skip_umap
    )
    if phase2_needed:
        print(f"\n[seed {args.seed}] phase 2 (metrics + optional plots, "
              f"UNTIMED) starting...")
        try:
            run_inference_and_analysis(
                run_dir=run_dir,
                silver_dir=args.silver_dir,
                label_keys=args.label_keys,
                skip_predict=True,                # already done in phase 1
                skip_code_index_plots=args.skip_code_index_plots,
                skip_svg_plots=args.skip_svg_plots,
                skip_umap=args.skip_umap,
                skip_metrics=args.skip_metrics,
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\nMETRICS / PLOTS FAILED on seed {args.seed}: "
                  f"{type(e).__name__}: {e}")
            print(tb)
            # Phase-1 runtime stamp is intentionally kept (training +
            # predict DID succeed; only metric/plot generation failed).
            # Status reflects the partial state so the aggregator can
            # decide whether to include this seed via `--seeds` override.
            _write_stamp(seed_runs_dir, args.seed, "status",
                         f"metrics_failed: {type(e).__name__}: {e}")
            return 4

    # Drop a sidecar describing what `runtime_seconds` measures, for
    # any future reader of this sweep dir. Same content for every seed;
    # writing per-seed keeps the file co-located with the stamp it
    # documents.
    _write_stamp(
        seed_runs_dir, args.seed, "runtime_methodology",
        "runtime_seconds covers train() + predict() forward / collation "
        "ONLY. EXCLUDED: the predicted_adata.h5ad disk write (stamped "
        "separately to predict_write_seconds.txt and subtracted from the "
        "wall-clock total here), metric computation "
        "(NMI/ARI/iLISI/MMD/Pearson), and any plot steps — all of those "
        "run in a second, untimed phase of _run_one_seed.py. For "
        "apples-to-apples comparison with baseline runners, those runners "
        "also exclude metric/plot cost and disk I/O from the per-seed "
        "timer (see `_record_seed_runtime` in "
        "analysis/benchmarking/cell_type_identification/run_pca_leiden.py "
        "for the shared convention), and add their shared embedding-compute "
        "cost back so the per-seed number reflects 'time to obtain "
        "clusters from scratch for this seed'.",
    )
    _write_stamp(seed_runs_dir, args.seed, "status", "OK")
    print(f"\n[seed {args.seed}] all done. runtime_seconds = {seconds:.1f}s "
          f"(train + predict compute only, AnnData write "
          f"{predict_write_seconds:.1f}s excluded); metrics + plots ran "
          "untimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
