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


# ---------------------------------------------------------------------------
# True data-split holdout eval (in-paint the cells SQUINT actually held out)
# ---------------------------------------------------------------------------
def _chunk_holdout(coords: np.ndarray, gidx: np.ndarray, chunk_size: int):
    """Greedily split held-out cells into spatially-contiguous chunks.

    Returns a list of global-index arrays, each of size <= ``chunk_size``.
    A big held-out region (> patch_size) must be filled chunk-by-chunk so each
    in-painting patch leaves room for observed context.
    """
    chunk_size = max(1, int(chunk_size))
    remaining = np.arange(gidx.size)
    chunks = []
    while remaining.size:
        seed = remaining[0]
        d2 = np.sum((coords[remaining] - coords[seed]) ** 2, axis=1)
        take = remaining[np.argsort(d2, kind="stable")[:chunk_size]]
        chunks.append(gidx[take])
        remaining = np.setdiff1d(remaining, take, assume_unique=False)
    return chunks


def _train_majority_baseline(source, eval_idx, train_idx, targets):
    """Most-frequent code in the TRAIN cells, scored on the held-out cells."""
    out = {}
    for (b, l) in targets:
        codes = source.codes[b]
        ref = codes[train_idx, l] if train_idx.size else codes[:, l]
        mode = np.bincount(ref).argmax()
        out[f"{b}__{l}"] = float((codes[eval_idx, l] == mode).mean())
    return out


def holdout_split_eval(
    model,
    source: AnnDataCodeSource,
    cfg: Stage2Config,
    chunk_size: int,
    device: Optional[str] = None,
    out_csv: Optional[str] = None,
    codes_out: Optional[str] = None,
    save_soft: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """In-paint the cells SQUINT actually held out (``source.holdout_mask``).

    Per section, the held-out cells are split into contiguous chunks of
    <= ``chunk_size``; each chunk is in-painted using ONLY that section's TRAIN
    cells as context (no other held-out cell leaks in), and the decoded top-1
    code accuracy is scored against the frozen true codes. Accuracies are
    aggregated cell-weighted across chunks. Baseline = most-frequent TRAIN code.
    """
    if not source.has_holdout:
        raise ValueError("source has no holdout mask; pass holdout_key/value")
    targets = cfg.prediction_targets
    rows_out: List[Dict[str, object]] = []
    all_holdout: List[np.ndarray] = []
    # predicted code stacks per held-out cell, accumulated for `codes_out`
    # (so the codes->expression decode step can read them without re-running).
    pred_idx_parts: List[np.ndarray] = []
    pred_code_parts: Dict[str, List[np.ndarray]] = {b.name: [] for b in source.branches}
    # per-(branch, level) SOFT distributions for the expected-embedding decode
    pred_prob_parts: Dict[tuple, List[np.ndarray]] = {(b, l): [] for (b, l) in cfg.prediction_targets}

    for s in source.section_ids:
        hrows = source.holdout_section_of(s)
        if hrows.size == 0:
            continue
        trows = source.train_section_of(s)
        chunks = _chunk_holdout(source.coords[hrows], hrows, chunk_size)
        if verbose:
            print(f"[eval] section {s}: {hrows.size} held-out cells, "
                  f"{trows.size} train context, {len(chunks)} chunk(s)")
        for ci, chunk in enumerate(chunks):
            if chunk.size < 1:
                continue
            res = inpaint(model, source, chunk, cfg, device=device,
                          observed_rows=trows, return_probs=save_soft)
            gidx = res["global_idx"]
            all_holdout.append(gidx)
            pred_idx_parts.append(gidx)
            for b in source.branches:
                pred_code_parts[b.name].append(np.asarray(res["codes"][b.name]))
            if save_soft and "probs" in res:
                for (b, l) in targets:
                    pred_prob_parts[(b, l)].append(np.asarray(res["probs"][b][l]))
            row: Dict[str, object] = {
                "section": int(s), "chunk": ci,
                "n_holdout": int(gidx.size), "patch_size": int(res["patch_size"]),
            }
            for (b, l) in targets:
                pred = res["codes"][b][:, l]
                gt = source.codes[b][gidx, l]
                row[f"acc_{b}__{l}"] = float((pred == gt).mean())
            rows_out.append(row)
            if verbose:
                acc0 = row.get(f"acc_{targets[0][0]}__{targets[0][1]}", float("nan"))
                print(f"[eval]   chunk {ci}: n={gidx.size}, patch={res['patch_size']}, "
                      f"acc({targets[0][0]} L{targets[0][1]})={acc0:.3f}")

    # persist predicted code stacks (global row index + per-branch codes) so
    # examples/stage2_decode_pearson.py can decode them through the frozen
    # stage-1 decoder without re-running the in-painting.
    if codes_out and pred_idx_parts:
        save_kw = {"global_idx": np.concatenate(pred_idx_parts).astype(np.int64)}
        for b in source.branches:
            save_kw[f"codes_{b.name}"] = np.concatenate(pred_code_parts[b.name]).astype(np.int64)
        if save_soft and all(pred_prob_parts[(b, l)] for (b, l) in targets):
            for (b, l) in targets:
                save_kw[f"probs_{b}_{l}"] = np.concatenate(
                    pred_prob_parts[(b, l)]).astype(np.float32)
        np.savez(codes_out, **save_kw)
        if verbose:
            print(f"[eval] wrote predicted codes{' + soft probs' if save_soft else ''} "
                  f"-> {codes_out}")

    # cell-weighted aggregation over chunks
    counts = np.array([r["n_holdout"] for r in rows_out], float)
    summary: Dict[str, object] = {
        "n_chunks_scored": len(rows_out),
        "n_holdout_total": int(counts.sum()) if counts.size else 0,
        "per_chunk": rows_out,
    }
    for (b, l) in targets:
        if rows_out and counts.sum() > 0:
            accs = np.array([r[f"acc_{b}__{l}"] for r in rows_out], float)
            summary[f"acc_{b}__{l}_mean"] = float(np.average(accs, weights=counts))
        else:
            summary[f"acc_{b}__{l}_mean"] = float("nan")
    summary["acc_overall_mean"] = float(
        np.nanmean([summary[f"acc_{b}__{l}_mean"] for (b, l) in targets])
    )

    eval_idx = np.unique(np.concatenate(all_holdout)) if all_holdout else np.array([], int)
    train_idx = np.unique(np.concatenate(
        [source.train_section_of(s) for s in source.section_ids]
    )) if source.section_ids.size else np.array([], int)
    base = (_train_majority_baseline(source, eval_idx, train_idx, targets)
            if eval_idx.size else {f"{b}__{l}": float("nan") for (b, l) in targets})
    for (b, l) in targets:
        summary[f"baseline_{b}__{l}"] = base.get(f"{b}__{l}", float("nan"))

    if out_csv and rows_out:
        _write_split_csv(out_csv, rows_out, summary, targets)
        if verbose:
            print(f"[eval] wrote {out_csv}")

    if verbose:
        print("[eval] === data-split holdout summary (decoded top-1 accuracy) ===")
        for (b, l) in targets:
            print(f"   {b} L{l}: {summary[f'acc_{b}__{l}_mean']:.3f}   "
                  f"(train-majority baseline {summary[f'baseline_{b}__{l}']:.3f})")
        print(f"   overall: {summary['acc_overall_mean']:.3f}  over "
              f"{summary['n_holdout_total']} held-out cells in {len(rows_out)} chunks")
    return summary


def _write_split_csv(path, rows, summary, targets):
    import csv

    target_cols = [f"acc_{b}__{l}" for (b, l) in targets]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "chunk", "n_holdout", "patch_size"] + target_cols)
        for row in rows:
            w.writerow([row["section"], row["chunk"], row["n_holdout"],
                        row["patch_size"]] + [row[c] for c in target_cols])
        w.writerow([])
        w.writerow(["WMEAN", "", summary["n_holdout_total"], ""] +
                   [summary[f"{c}_mean"] for c in target_cols])
        w.writerow(["BASELINE", "", "", ""] +
                   [summary[f"baseline_{b}__{l}"] for (b, l) in targets])
