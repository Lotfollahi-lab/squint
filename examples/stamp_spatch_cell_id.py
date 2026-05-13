#!/usr/bin/env python3
"""
Stamp ``adata.obs['cell_id']`` on every ``dataset_id_<N>.h5ad`` in the
spatch_1p silver folder so ``InMemoryDatasetBlob.process()`` (line 522:
``batch_dict['cell_id'] = adata_batch.obs['cell_id'].to_list()``) can
collect per-cell identifiers without a ``KeyError``.

Format
------
For each AnnData with ``uns['batch'] = 'batchN'`` (or ``N``), this
script stamps:

    obs['cell_id'] = f"spatch_batch{N}_{obs_name}"     for each row

The prefix guarantees global uniqueness across sections even when
``adata.obs_names`` collide across sections (they're typically unique
within a section because AnnData enforces it, but two sections can
both use ``"0", "1", "2", ..."``).

This is what the blob loader actually uses ``cell_id`` for:
  1. Per-cell metadata propagation through training (saved into
     ``batch_dict['cell_id']``, then surfaced at inference time so
     each predicted-AnnData row can be tied back to its source).
  2. Fallback batch-one-hot parsing — the loader can parse
     ``..._batchN_...`` substrings out of ``cell_id`` when
     ``obs['batch']`` is missing. We already stamped
     ``uns['batch']`` via ``stamp_spatch_uns_batch.py``, so this
     fallback won't fire for spatch_1p, but the chosen format is
     compatible with it anyway.

Prerequisites
-------------
  - Every file must already have ``uns['batch']`` set (run
    ``examples/stamp_spatch_uns_batch.py`` first). This script needs
    ``uns['batch']`` to derive the integer ``N`` for the prefix.

Usage
-----
    python examples/stamp_spatch_cell_id.py
    # or, to inspect without writing:
    python examples/stamp_spatch_cell_id.py --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad

SPATCH_DIR = Path("/nfs/team361/sb75/DATASETS/silver/spatch_1p")
FILENAME_PATTERN = re.compile(r"^dataset_id_(\d+)\.h5ad$")


def _derive_batch_int(uns_batch) -> int:
    """
    Same parsing rules as
    ``InMemoryDatasetBlob._derive_adata_batch_id``: int /
    int-string / 'batchN' string -> int.
    """
    # int / numpy int-like
    try:
        import numpy as _np
        if isinstance(uns_batch, (int, _np.integer)):
            return int(uns_batch)
    except ImportError:
        if isinstance(uns_batch, int):
            return int(uns_batch)
    if isinstance(uns_batch, str):
        if uns_batch.startswith("batch"):
            try:
                return int(uns_batch[5:])
            except ValueError:
                pass
        try:
            return int(uns_batch)
        except ValueError:
            pass
    raise ValueError(f"Could not parse uns['batch']={uns_batch!r} as int.")


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
        if not FILENAME_PATTERN.match(h5ad_path.name):
            print(f"SKIP  {h5ad_path.name} — does not match dataset_id_<N>.h5ad")
            continue

        adata = ad.read_h5ad(h5ad_path)

        uns_batch = adata.uns.get("batch", None)
        if uns_batch is None:
            print(
                f"SKIP  {h5ad_path.name}  uns['batch'] missing — run "
                "examples/stamp_spatch_uns_batch.py first"
            )
            continue
        try:
            batch_int = _derive_batch_int(uns_batch)
        except ValueError as e:
            print(f"SKIP  {h5ad_path.name}  {e}")
            continue

        prefix = f"spatch_batch{batch_int}_"
        new_cell_ids = [f"{prefix}{name}" for name in adata.obs_names.astype(str)]

        # Idempotency: if every cell_id already starts with the prefix
        # we'd generate, don't rewrite.
        existing = adata.obs.get("cell_id", None)
        if existing is not None and all(
            isinstance(v, str) and v.startswith(prefix) for v in existing
        ) and len(existing) == len(new_cell_ids) and list(existing) == new_cell_ids:
            print(
                f"OK    {h5ad_path.name:25s}  obs['cell_id'] already "
                f"stamped (prefix={prefix!r}) — skipping"
            )
            continue

        n = len(new_cell_ids)
        sample = new_cell_ids[0] if n > 0 else "(empty)"
        print(
            f"STAMP {h5ad_path.name:25s}  obs['cell_id'] <- "
            f"{n} entries prefixed {prefix!r}  (e.g. {sample!r})"
        )
        if not dry_run:
            adata.obs["cell_id"] = new_cell_ids
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
