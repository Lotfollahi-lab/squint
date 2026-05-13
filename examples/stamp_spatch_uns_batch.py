#!/usr/bin/env python3
"""
Stamp ``adata.uns['batch']`` on every ``dataset_id_<N>.h5ad`` in the
spatch_1p silver folder so ``InMemoryDatasetBlob._derive_adata_batch_id``
can derive the integer ``adata_batch_id`` per section at blob-build
time.

Format note
-----------
The blob loader accepts ``uns['batch']`` in three shapes:
  * int (``2``, ``np.int64(2)``)
  * int-string (``'2'``)
  * ``'batchN'`` string with **no separator** (``'batch2'``)

``'batch_N'`` with an underscore is *not* parseable (the loader strips
``'batch'`` and then calls ``int('_2')`` which raises). This script
uses ``f'batch{N}'`` so the spatch_1p smoke-test variant
(``smoke-test+spatch_1p``, which expects ``adata_batch_id == 2`` for
``dataset_id_2.h5ad``) Just Works.

Usage
-----
    python examples/stamp_spatch_uns_batch.py
    # or, to inspect without writing:
    python examples/stamp_spatch_uns_batch.py --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad

SPATCH_DIR = Path("/nfs/team361/sb75/DATASETS/silver/spatch_1p")
PATTERN = re.compile(r"^dataset_id_(\d+)\.h5ad$")


def main(spatch_dir: Path, dry_run: bool) -> None:
    if not spatch_dir.is_dir():
        raise SystemExit(f"Not a directory: {spatch_dir}")

    h5ad_paths = sorted(spatch_dir.glob("dataset_id_*.h5ad"))
    if not h5ad_paths:
        raise SystemExit(f"No dataset_id_*.h5ad files found under {spatch_dir}")

    print(f"Found {len(h5ad_paths)} file(s) under {spatch_dir}")
    print(f"Dry run: {dry_run}")
    print()

    for h5ad_path in h5ad_paths:
        m = PATTERN.match(h5ad_path.name)
        if m is None:
            # Skip non-matching filenames (e.g. extract.log lives in the
            # same dir but won't match the glob; still a defensive guard
            # if a future file shows up that doesn't fit the convention).
            print(f"SKIP  {h5ad_path.name} — does not match dataset_id_<N>.h5ad")
            continue

        dataset_id = int(m.group(1))
        new_batch = f"batch{dataset_id}"   # blob-loader-compatible form

        # Open in r+ to avoid a full read+write rewrite where possible;
        # AnnData's append-mode handling of `uns` is reliable across
        # versions for scalar strings, but to keep this script
        # bullet-proof we just do a plain read -> set -> write_h5ad.
        adata = ad.read_h5ad(h5ad_path)
        prev = adata.uns.get("batch", None)

        if prev == new_batch:
            print(
                f"OK    {h5ad_path.name:25s}  uns['batch'] already "
                f"{new_batch!r} — skipping"
            )
            continue

        print(
            f"STAMP {h5ad_path.name:25s}  uns['batch'] "
            f"{prev!r} -> {new_batch!r}"
        )
        if not dry_run:
            adata.uns["batch"] = new_batch
            adata.write_h5ad(h5ad_path, compression="gzip")

    print()
    print("done." if not dry_run else "done (dry-run — nothing written).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--spatch-dir",
        type=Path,
        default=SPATCH_DIR,
        help=f"Directory containing the spatch_1p dataset_id_*.h5ad files "
             f"(default: {SPATCH_DIR}).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files.",
    )
    args = ap.parse_args()
    main(args.spatch_dir, args.dry_run)
