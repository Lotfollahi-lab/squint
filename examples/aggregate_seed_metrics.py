#!/usr/bin/env python3
"""
Re-aggregate per-seed metric CSVs into the multi-seed summary tables — a
minimal, dependency-light standalone (pandas only; NO stamps / manifest / torch
required, unlike run_squint_multi_seed.py).

Use case
--------
You recomputed a metric on the individual seeds AFTER the multi-seed run (e.g.
added MMD to each seed's `batch_integration_metrics.csv` by re-running inference
metrics-only), but the summary `metrics/per_seed_*.csv` that the plots read is
stale. This script rebuilds those summaries from whatever the per-seed files
currently hold — so the freshly-added metric (MMD) lands in the summary.

What it does
------------
Under ``--sweep-dir`` it globs every per-seed metric CSV (searching the whole
tree, EXCLUDING the sweep's own top-level ``metrics/`` so the aggregates aren't
re-ingested), infers each file's seed from its path (``seed_<N>`` / ``seedN`` /
``_seed<N>``; latest timestamp wins if a seed has several), prepends a ``seed``
column, concatenates, and writes the summary into ``<sweep-dir>/metrics/``:

    batch_integration_metrics.csv     -> per_seed_batch_integration.csv
    niche_identification_metrics.csv  -> per_seed_niche_identification.csv
    pearson_reconstruction_metrics.csv-> per_seed_pearson_reconstruction.csv

Idempotent — overwrites the summaries. The per-seed files are never modified.

Usage
-----
  python examples/aggregate_seed_metrics.py --sweep-dir <.../<variant>__multiseed/<TS>>

  # only the batch-integration table (the MMD case):
  python examples/aggregate_seed_metrics.py --sweep-dir <...> --tables batch_integration_metrics.csv

  # per-seed runs live somewhere other than under the sweep dir:
  python examples/aggregate_seed_metrics.py --sweep-dir <...> --search-root <dir-holding-the-seed-runs>

  --dry-run  : list what would be aggregated (seed -> file) without writing.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# per-seed filename -> summary filename (same mapping as
# run_squint_multi_seed._AGGREGATED_TABLES).
_TABLE_MAP: Dict[str, str] = {
    "batch_integration_metrics.csv":      "per_seed_batch_integration.csv",
    "niche_identification_metrics.csv":   "per_seed_niche_identification.csv",
    "pearson_reconstruction_metrics.csv": "per_seed_pearson_reconstruction.csv",
}

_SEED_RE = re.compile(r"seed_?(\d+)")


def _infer_seed(rel_path: str) -> Optional[int]:
    """Seed from a path like `seed_runs/seed_2/<TS>/metrics/...` or
    `<TS>_seed0/metrics/...`. Takes the LAST `seed_?<N>` match (deepest /
    most specific). `seed_runs` alone has no digit so it never matches."""
    m = _SEED_RE.findall(rel_path)
    return int(m[-1]) if m else None


def _discover(search_root: Path, sweep_metrics: Path, fname: str
              ) -> Dict[int, Path]:
    """seed -> newest per-seed CSV named `fname` under `search_root`, skipping
    the sweep's own metrics/ dir. 'Newest' = lexicographically-largest path
    (timestamps sort as calendar order)."""
    seed_to_file: Dict[int, Path] = {}
    # sorted() so a later timestamp for the same seed overwrites the earlier.
    for f in sorted(search_root.rglob(f"*/metrics/{fname}")) + \
             sorted(search_root.rglob(f"metrics/{fname}")):
        if f.parent.resolve() == sweep_metrics.resolve():
            continue  # this IS the aggregate location — never ingest it
        rel = str(f.relative_to(search_root))
        seed = _infer_seed(rel)
        if seed is None:
            print(f"  [skip] no seed in path: {rel}", file=sys.stderr)
            continue
        seed_to_file[seed] = f    # later (sorted) path for a seed wins
    return dict(sorted(seed_to_file.items()))


def _aggregate_one(seed_to_file: Dict[int, Path], out_csv: Path,
                   dry_run: bool) -> int:
    frames: List[pd.DataFrame] = []
    for seed, f in seed_to_file.items():
        try:
            df = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [seed {seed}] failed to read {f}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        # If the per-seed file already carries a 'seed' column (unusual for a
        # single run), trust the path-derived seed and overwrite it.
        df = df.drop(columns=["seed"], errors="ignore")
        df.insert(0, "seed", int(seed))
        frames.append(df)
    if not frames:
        print(f"  (no usable per-seed files -> not writing {out_csv.name})")
        return 0
    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"  seeds {sorted(seed_to_file)} -> {out_csv}  "
          f"({len(out)} rows over {len(frames)} seed(s))")
    if not dry_run:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
    return len(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="Multi-seed run dir; summaries are written to "
                        "<sweep-dir>/metrics/per_seed_*.csv.")
    p.add_argument("--search-root", type=Path, default=None,
                   help="Where to look for the per-seed metric CSVs (default: "
                        "--sweep-dir). Set this if the per-seed runs live "
                        "outside the sweep dir.")
    p.add_argument("--tables", type=str, default=",".join(_TABLE_MAP),
                   help="Comma-separated per-seed filenames to aggregate. "
                        f"Default: all of {list(_TABLE_MAP)}.")
    p.add_argument("--dry-run", action="store_true",
                   help="List seed->file mapping without writing.")
    args = p.parse_args(argv)

    if not args.sweep_dir.is_dir():
        raise SystemExit(f"--sweep-dir not found: {args.sweep_dir}")
    search_root = args.search_root or args.sweep_dir
    if not search_root.is_dir():
        raise SystemExit(f"--search-root not found: {search_root}")
    sweep_metrics = args.sweep_dir / "metrics"

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = [t for t in tables if t not in _TABLE_MAP]
    if unknown:
        raise SystemExit(f"unknown --tables entries {unknown}; "
                         f"valid: {list(_TABLE_MAP)}")

    print(f"Sweep dir   : {args.sweep_dir}")
    print(f"Search root : {search_root}")
    print(f"Summaries -> {sweep_metrics}")
    print(f"Tables      : {tables}")
    print(f"Dry run     : {args.dry_run}\n")

    total = 0
    for fname in tables:
        print(f"[{fname}]")
        seed_to_file = _discover(search_root, sweep_metrics, fname)
        if not seed_to_file:
            print(f"  no per-seed '{fname}' found under {search_root} "
                  f"(outside {sweep_metrics}).")
            continue
        total += _aggregate_one(
            seed_to_file, sweep_metrics / _TABLE_MAP[fname], args.dry_run)
        print()

    if total == 0:
        print("Nothing aggregated. Check that the per-seed CSVs exist under "
              "--search-root and that their paths contain a seed_<N> token.")
        return 1
    print(f"DONE{' (dry-run)' if args.dry_run else ''}. "
          f"{total} rows written across {len(tables)} table(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
