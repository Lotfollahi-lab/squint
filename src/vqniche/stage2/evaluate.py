"""
Region-holdout in-painting evaluation for the stage-2 prior (torch + numpy).

Carves contiguous spatial holes out of the frozen ``predicted_adata``, in-paints
each with the model, and scores the predicted code stacks against the true codes
(which we have, since stage 1 assigns a code to every cell). Reports decoded
top-1 accuracy per (branch, level), plus a majority-code baseline for context.

This is the metric that matches the downstream task ("can stage 2 fill a hole in
the tissue from spatial context alone?") and uses the iterative decoder, not
teacher forcing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import Stage2Config
from .data import AnnDataCodeSource
from .masking import block_mask
from .decode import inpaint


def _majority_baseline_acc(source: AnnDataCodeSource, gidx: np.ndarray) -> Dict[str, float]:
    """Accuracy of always predicting each level's global most-frequent code."""
    out: Dict[str, float] = {}
    for b in source.branches:
        codes = source.codes[b.name]
        for l in range(b.num_levels):
            mode = np.bincount(codes[:, l]).argmax()
            out[f"{b.name}__{l}"] = float((codes[gidx, l] == mode).mean())
    return out


def region_holdout_eval(
    model,
    source: AnnDataCodeSource,
    cfg: Stage2Config,
    n_regions: int = 8,
    holdout_frac: float = 0.1,
    seed: int = 0,
    device: Optional[str] = None,
    out_csv: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """In-paint ``n_regions`` contiguous holes and score code recovery.

    Returns a summary dict with per-target mean/std accuracy (over regions),
    the majority baseline, and the per-region rows.
    """
    rng = np.random.default_rng(seed)
    targets = cfg.prediction_targets

    # sample sections weighted by size
    counts = np.array([source.section_of(s).size for s in source.section_ids], float)
    sec_probs = counts / counts.sum()

    rows: List[Dict[str, object]] = []
    all_holdout: List[np.ndarray] = []
    for r in range(n_regions):
        s = int(source.section_ids[rng.choice(len(sec_probs), p=sec_probs)])
        srows = source.section_of(s)
        sub = block_mask(source.coords[srows], holdout_frac, rng)
        holdout = srows[sub]
        if holdout.size < 2:
            continue

        res = inpaint(model, source, holdout, cfg, device=device)
        gidx = res["global_idx"]                            # decoder's order
        all_holdout.append(gidx)

        row: Dict[str, object] = {
            "region": r, "section": s,
            "n_holdout": int(gidx.size), "patch_size": int(res["patch_size"]),
        }
        for ti, (b, l) in enumerate(targets):
            pred = res["codes"][b][:, l]
            gt = source.codes[b][gidx, l]
            row[f"acc_{b}__{l}"] = float((pred == gt).mean())
        rows.append(row)
        if verbose:
            acc0 = row.get(f"acc_{targets[0][0]}__{targets[0][1]}", float("nan"))
            print(f"[eval] region {r}: section {s}, holdout {gidx.size}, "
                  f"patch {res['patch_size']}, acc({targets[0][0]} L{targets[0][1]})={acc0:.3f}")

    # aggregate
    summary: Dict[str, object] = {"n_regions_scored": len(rows), "per_region": rows}
    for ti, (b, l) in enumerate(targets):
        vals = np.array([row[f"acc_{b}__{l}"] for row in rows], float) if rows else np.array([np.nan])
        summary[f"acc_{b}__{l}_mean"] = float(np.nanmean(vals))
        summary[f"acc_{b}__{l}_std"] = float(np.nanstd(vals))
    # overall mean across all targets
    target_means = [summary[f"acc_{b}__{l}_mean"] for (b, l) in targets]
    summary["acc_overall_mean"] = float(np.nanmean(target_means))

    # majority-code baseline over the union of actually-held-out cells
    union = np.unique(np.concatenate(all_holdout)) if all_holdout else np.array([], dtype=int)
    base = (_majority_baseline_acc(source, union) if union.size
            else {f"{b}__{l}": float("nan") for (b, l) in targets})
    for (b, l) in targets:
        summary[f"baseline_{b}__{l}"] = base.get(f"{b}__{l}", float("nan"))

    if out_csv and rows:
        _write_csv(out_csv, rows, summary, targets)
        if verbose:
            print(f"[eval] wrote {out_csv}")

    if verbose:
        print("[eval] === summary (decoded top-1 accuracy) ===")
        for (b, l) in targets:
            print(f"   {b} L{l}: {summary[f'acc_{b}__{l}_mean']:.3f} "
                  f"± {summary[f'acc_{b}__{l}_std']:.3f}   "
                  f"(majority baseline {summary[f'baseline_{b}__{l}']:.3f})")
        print(f"   overall: {summary['acc_overall_mean']:.3f}  "
              f"over {len(rows)} regions")
    return summary


def _write_csv(path, rows, summary, targets):
    import csv

    target_cols = [f"acc_{b}__{l}" for (b, l) in targets]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "section", "n_holdout", "patch_size"] + target_cols)
        for row in rows:
            w.writerow([row["region"], row["section"], row["n_holdout"],
                        row["patch_size"]] + [row[c] for c in target_cols])
        w.writerow([])
        w.writerow(["MEAN", "", "", ""] + [summary[f"{c}_mean"] for c in target_cols])
        w.writerow(["STD", "", "", ""] + [summary[f"{c}_std"] for c in target_cols])
        w.writerow(["BASELINE", "", "", ""] +
                   [summary[f"baseline_{b}__{l}"] for (b, l) in targets])
