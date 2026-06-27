#!/usr/bin/env python3
"""
Aggregator for the multi-seed SQUINT submission. Reads the per-seed
stamp files written by `_run_one_seed.py` and the metric CSVs in each
seed's SQUINT run dir, then concatenates them into the long-format
CSVs that the benchmark figure scripts (`plot_niche_identification_
benchmark.py`, `plot_cell_type_identification_benchmark.py`) consume.

Designed to be run AFTER all per-seed LSF jobs finish — typically as
a final dependent job chained via `bsub -w 'done(seed_0) && ...'` by
`submit_multi_seed.sh`. Also usable standalone for re-aggregating
existing per-seed runs (e.g. after a metrics-only bugfix).

Input layout (created by `submit_multi_seed.sh` + `_run_one_seed.py`):

    <OUT_DIR>/                                         # sweep dir
        manifest.yaml                                  # variant, seeds, ...
        seed_runs/
            seed_0_run_dir.txt                         # SQUINT run dir
            seed_0_runtime_seconds.txt                 # train + predict only (apples-to-apples)
            seed_0_runtime_methodology.txt             # explainer of what runtime covers
            seed_0_status.txt                          # "OK" or error msg
            seed_1_run_dir.txt
            ...

Output layout (written into <OUT_DIR>/):

    <OUT_DIR>/
        metrics/
            per_seed_niche_identification.csv          (1 row/seed × code × label)
            per_seed_batch_integration.csv             (1 row/seed × emb × metric)
            per_seed_pearson_reconstruction.csv        (1 row/seed × branch × axis × ...)
            per_seed_runtimes.csv                      (1 row per seed)
            runtime_summary.csv                        (mean / std / min / max / total)
        seed_run_index.csv                             (seed → run_dir mapping + status)

Usage:

    # Default: read <OUT_DIR>/seed_runs/, aggregate, write into <OUT_DIR>/.
    python examples/run_squint_multi_seed.py --out-dir <OUT_DIR>

    # Method label stamped into runtime CSVs (default: variant name from
    # manifest.yaml, falling back to the sweep dir's parent name):
    python examples/run_squint_multi_seed.py --out-dir <OUT_DIR> \\
        --method-label SQUINT

    # Override which seeds to include (default: all that have an
    # `seed_<N>_status.txt == "OK"` stamp):
    python examples/run_squint_multi_seed.py --out-dir <OUT_DIR> --seeds 0,1,2,4
"""
from __future__ import annotations

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

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Stamp-file discovery
# ---------------------------------------------------------------------------

_STAMP_RE = re.compile(
    r"^seed_(\d+)_(run_dir|runtime_seconds|runtime_methodology|status)\.txt$"
)


def _discover_seed_stamps(
        seed_runs_dir: Path,
    ) -> Dict[int, Dict[str, str]]:
    """Walk `<OUT_DIR>/seed_runs/` and collect the stamp values per seed.

    Returns: {seed: {"run_dir": "...", "runtime_seconds": "1234.5", "status": "OK"}}
    Missing stamps are simply absent from the inner dict (callers must
    handle that)."""
    out: Dict[int, Dict[str, str]] = {}
    if not seed_runs_dir.is_dir():
        return out
    for f in sorted(seed_runs_dir.iterdir()):
        m = _STAMP_RE.match(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        key = m.group(2)
        try:
            out.setdefault(seed, {})[key] = f.read_text().strip()
        except OSError as exc:
            print(f"  WARN: could not read {f}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Per-seed CSV reading
# ---------------------------------------------------------------------------

def _read_metric_csv(
        run_dir: Path,
        relpath: str,
        seed: int,
    ) -> Optional[pd.DataFrame]:
    """Read `run_dir/<relpath>` if present; prepend a `seed` column."""
    csv_path = run_dir / relpath
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    df.insert(0, "seed", int(seed))
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# (metric_filename_under_seed_run/<run_dir>/metrics/, aggregated_output_filename)
_AGGREGATED_TABLES: List[Tuple[str, str]] = [
    ("metrics/niche_identification_metrics.csv",
     "per_seed_niche_identification.csv"),
    ("metrics/batch_integration_metrics.csv",
     "per_seed_batch_integration.csv"),
    ("metrics/pearson_reconstruction_metrics.csv",
     "per_seed_pearson_reconstruction.csv"),
]


def _aggregate(
        seed_to_run_dir: Dict[int, Path],
        seed_to_runtime: Dict[int, Optional[float]],
        seed_to_status: Dict[int, str],
        out_dir: Path,
        method_label: str,
    ) -> None:
    """Concatenate per-seed metric CSVs into long-format aggregates and
    write the runtime CSVs. Idempotent — overwrites if rerun."""
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAggregating metrics for {len(seed_to_run_dir)} seed(s):")
    for src_rel, dst_name in _AGGREGATED_TABLES:
        frames: List[pd.DataFrame] = []
        for seed, run_dir in sorted(seed_to_run_dir.items()):
            df = _read_metric_csv(run_dir, src_rel, seed)
            if df is None:
                print(f"  [seed {seed}] missing {src_rel} -> "
                      f"{run_dir / src_rel}")
                continue
            frames.append(df)
        if not frames:
            print(f"  (no seed produced {src_rel}; skipping {dst_name})")
            continue
        long_df = pd.concat(frames, ignore_index=True, sort=False)
        out = metrics_dir / dst_name
        long_df.to_csv(out, index=False)
        print(f"  -> {out}  ({len(long_df)} rows over {len(frames)} seeds)")

    # Runtime CSVs (same column convention as the baseline runners,
    # e.g. analysis/benchmarking/cell_type_identification/run_pca_leiden.py).
    rt_rows = []
    for seed, secs in sorted(seed_to_runtime.items()):
        rt_rows.append({
            "seed": int(seed),
            "method": method_label,
            "runtime_seconds": (float(secs) if secs is not None else float("nan")),
        })
    if rt_rows:
        rt_df = pd.DataFrame(rt_rows)
        rt_path = metrics_dir / "per_seed_runtimes.csv"
        rt_df.to_csv(rt_path, index=False)
        print(f"  -> {rt_path}")

        secs = rt_df["runtime_seconds"].astype(float).dropna()
        if not secs.empty:
            summary = pd.DataFrame([{
                "method":        method_label,
                "n_seeds":       int(len(secs)),
                "mean_seconds":  float(secs.mean()),
                "std_seconds":   float(secs.std(ddof=1)) if len(secs) > 1 else 0.0,
                "min_seconds":   float(secs.min()),
                "max_seconds":   float(secs.max()),
                "total_seconds": float(secs.sum()),
                # Bake the methodology into the summary so it travels
                # with the CSV. Cross-method runtime comparisons live or
                # die on getting this right.
                "runtime_includes":
                    "train + inference (predict); excludes metric "
                    "computation, UMAP, and downstream plots — the "
                    "produced codes ARE the clusters",
            }])
            out = metrics_dir / "runtime_summary.csv"
            summary.to_csv(out, index=False)
            print(f"  -> {out}  (mean={secs.mean():.1f}s ± "
                  f"{secs.std(ddof=1) if len(secs) > 1 else 0.0:.1f}s, "
                  f"total={secs.sum():.1f}s)")

    # Seed → run_dir + status index.
    idx_rows = []
    for seed in sorted(set(seed_to_run_dir) | set(seed_to_status)):
        idx_rows.append({
            "seed": int(seed),
            "run_dir": str(seed_to_run_dir.get(seed, "")),
            "status": seed_to_status.get(seed, ""),
            "runtime_seconds": (
                float(seed_to_runtime.get(seed, float("nan")))
                if seed_to_runtime.get(seed) is not None else float("nan")
            ),
        })
    if idx_rows:
        out = out_dir / "seed_run_index.csv"
        pd.DataFrame(idx_rows).to_csv(out, index=False)
        print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _read_manifest(out_dir: Path) -> Dict:
    """Parse the manifest.yaml written by `submit_multi_seed.sh`.
    Returns {} if missing or unparseable — callers must handle missing
    fields gracefully."""
    path = out_dir / "manifest.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: could not parse {path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Sweep dir created by `submit_multi_seed.sh`. Expected to "
             "contain seed_runs/ (with seed_<N>_*.txt stamp files) and "
             "optionally manifest.yaml. Aggregated CSVs are written into "
             "<OUT_DIR>/metrics/.",
    )
    p.add_argument(
        "--method-label", type=str, default=None,
        help="Label stamped into per_seed_runtimes.csv + runtime_summary.csv. "
             "Default: variant name from <OUT_DIR>/manifest.yaml, falling "
             "back to <OUT_DIR>'s grandparent (the variant slug under "
             "<dataset>/).",
    )
    p.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed list to include. Default: every seed "
             "with `seed_<N>_status.txt == 'OK'` (i.e. only successful "
             "runs). Pass an explicit list to include failed seeds too.",
    )
    args = p.parse_args()

    if not args.out_dir.is_dir():
        raise SystemExit(f"OUT_DIR does not exist: {args.out_dir}")
    seed_runs_dir = args.out_dir / "seed_runs"
    if not seed_runs_dir.is_dir():
        raise SystemExit(
            f"No seed_runs/ subdir under {args.out_dir}. "
            f"Either this isn't a multi-seed sweep dir, or the seed jobs "
            f"haven't written any stamps yet."
        )

    # Discover stamps
    stamps = _discover_seed_stamps(seed_runs_dir)
    if not stamps:
        raise SystemExit(
            f"No seed stamps found under {seed_runs_dir}. "
            f"Have any of the per-seed LSF jobs completed?"
        )

    # Resolve method label
    manifest = _read_manifest(args.out_dir)
    method_label = (
        args.method_label
        or manifest.get("variant")
        or args.out_dir.resolve().parent.parent.name
    )

    # Filter seeds
    if args.seeds:
        keep = set(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    else:
        keep = {s for s, info in stamps.items()
                if info.get("status", "").startswith("OK")}

    print(f"Sweep dir   : {args.out_dir}")
    print(f"Method label: {method_label}")
    print(f"Seeds found : {sorted(stamps)}")
    print(f"Seeds kept  : {sorted(keep)}")

    seed_to_run_dir: Dict[int, Path] = {}
    seed_to_runtime: Dict[int, Optional[float]] = {}
    seed_to_status: Dict[int, str] = {}
    for seed, info in stamps.items():
        seed_to_status[seed] = info.get("status", "")
        if seed not in keep:
            continue
        run_dir_str = info.get("run_dir")
        if not run_dir_str:
            print(f"  [seed {seed}] no run_dir stamp — skipping")
            continue
        run_dir = Path(run_dir_str)
        if not run_dir.is_dir():
            print(f"  [seed {seed}] run_dir does not exist: {run_dir}; skipping")
            continue
        seed_to_run_dir[seed] = run_dir
        rt = info.get("runtime_seconds")
        try:
            seed_to_runtime[seed] = float(rt) if rt else None
        except ValueError:
            seed_to_runtime[seed] = None

    if not seed_to_run_dir:
        raise SystemExit(
            "No usable seed runs to aggregate. Check stamp files in "
            f"{seed_runs_dir} and the run_dir paths they point at."
        )

    # GUARD: detect seeds that resolved to the SAME run dir. This happens
    # when parallel seed jobs collide on a seconds-granularity timestamp
    # (fixed in train() by appending `_seed<N>`; older sweeps or any
    # regression surface here). Such seeds are NOT independent — the
    # per_seed_*.csv would carry duplicate rows and the benchmark figures
    # would show falsely tight error bars.
    _by_dir: Dict[str, List[int]] = {}
    for _seed, _rd in seed_to_run_dir.items():
        _by_dir.setdefault(str(_rd), []).append(_seed)
    _collisions = {d: sorted(ss) for d, ss in _by_dir.items() if len(ss) > 1}
    if _collisions:
        print("\n" + "!" * 78)
        print("WARNING: multiple seeds resolved to the SAME run dir — they are "
              "NOT independent runs:")
        for _d, _ss in sorted(_collisions.items()):
            print(f"  seeds {_ss} -> {_d}")
        print(f"  => {len(seed_to_run_dir)} seed stamps map to only "
              f"{len(_by_dir)} distinct run dir(s); per_seed_*.csv will contain "
              f"DUPLICATE rows and any mean/SEM over seeds will be biased "
              f"(falsely tight error bars).")
        print("  Fix: rerun the multi-seed sweep with the train() seed-suffix "
              "fix so each seed gets its own `<TS>_seed<N>` dir.")
        print("!" * 78 + "\n")

    _aggregate(
        seed_to_run_dir=seed_to_run_dir,
        seed_to_runtime=seed_to_runtime,
        seed_to_status=seed_to_status,
        out_dir=args.out_dir,
        method_label=method_label,
    )

    print()
    print("=" * 78)
    print(f"Aggregation complete.")
    print(f"  sweep dir      : {args.out_dir}")
    print(f"  method label   : {method_label}")
    print(f"  seeds aggregated: {sorted(seed_to_run_dir)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
