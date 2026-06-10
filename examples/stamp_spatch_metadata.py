#!/usr/bin/env python3
"""
Stamp ``adata.uns['dataset_id']``, ``adata.uns['tissue']``, and
``adata.uns['species']`` on every ``dataset_id_<N>.h5ad`` in the
spatch_1p silver folder.

These three keys are pure metadata: ``InMemoryDatasetBlob`` stores
them on each PyG ``Data`` object and round-trips them back to
``predicted_adata.uns`` at inference time for traceability, but no
training / loss / metric code reads them. ``in_memory_dataset_blob.py``
already supplies safe defaults if these keys are missing
(``dataset_id -> self.name``, ``tissue/species -> 'unknown'``), so
this script is only needed if you want explicit, non-default values
in the predicted AnnDatas.

For spatch_1p the user-specified values are:
    uns['dataset_id'] = 3000
    uns['tissue']     = 'ovary'
    uns['species']    = 'human'

Usage
-----
    python examples/stamp_spatch_metadata.py
    # or, to inspect without writing:
    python examples/stamp_spatch_metadata.py --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad

SPATCH_DIR = Path("/nfs/team361/sb75/DATASETS/silver/spatch_1p")
FILENAME_PATTERN = re.compile(r"^dataset_id_(\d+)\.h5ad$")

# Values to stamp — adjust here if you want different metadata.
DATASET_ID = 3000
TISSUE = "ovary"
SPECIES = "human"


def main(spatch_dir: Path, dry_run: bool) -> None:
    if not spatch_dir.is_dir():
        raise SystemExit(f"Not a directory: {spatch_dir}")

    h5ad_paths = sorted(spatch_dir.glob("dataset_id_*.h5ad"))
    if not h5ad_paths:
        raise SystemExit(f"No dataset_id_*.h5ad files found under {spatch_dir}")

    print(f"Found {len(h5ad_paths)} file(s) under {spatch_dir}")
    print(f"Will stamp: uns['dataset_id']={DATASET_ID!r}, "
          f"uns['tissue']={TISSUE!r}, uns['species']={SPECIES!r}")
    print(f"Dry run: {dry_run}")
    print()

    for h5ad_path in h5ad_paths:
        if not FILENAME_PATTERN.match(h5ad_path.name):
            print(f"SKIP  {h5ad_path.name} — does not match dataset_id_<N>.h5ad")
            continue

        adata = ad.read_h5ad(h5ad_path)

        prev = {
            "dataset_id": adata.uns.get("dataset_id", None),
            "tissue":     adata.uns.get("tissue", None),
            "species":    adata.uns.get("species", None),
        }
        already = (
            prev["dataset_id"] == DATASET_ID
            and prev["tissue"] == TISSUE
            and prev["species"] == SPECIES
        )
        if already:
            print(
                f"OK    {h5ad_path.name:25s}  already stamped — skipping"
            )
            continue

        print(
            f"STAMP {h5ad_path.name:25s}  "
            f"dataset_id {prev['dataset_id']!r} -> {DATASET_ID!r}, "
            f"tissue {prev['tissue']!r} -> {TISSUE!r}, "
            f"species {prev['species']!r} -> {SPECIES!r}"
        )

        if not dry_run:
            adata.uns["dataset_id"] = DATASET_ID
            adata.uns["tissue"] = TISSUE
            adata.uns["species"] = SPECIES
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
