#!/usr/bin/env python
"""
Read-only leakage audit for the held-out-region (region-holdout) protocol.

Reads one or two stage-1 ``predicted_adata.h5ad`` files and reports:

  * the number of edges in ``uns['edge_index']`` that go from a held-out
    source to an observed target (PyG convention: row 0 = src, row 1 = dst,
    messages flow src -> dst). These are the edges through which held-out
    expression reaches observed cells. It must be 0 for an inductive run
    (``holdout_graph`` 'rebuild' / 'drop').
  * the observed boundary band (observed cells with >= 1 held-out
    in-neighbour) and the held-out ring (held-out cells with >= 1 observed
    out-neighbour). On an old transductive run (--pred-ref) these measure
    the exposure.
  * edges INTO held-out cells by source type (shows holdout_test_inedges)
    and the self-loop count per node.
  * n_test per file, and whether obs_names / the held-out split are
    identical between --pred and --pred-ref (row for row).
  * holdout provenance stamped by predict() into ``uns['squint']`` (or the
    legacy ``uns['vqniche']``; values may be stringified, parsed safely).
  * the checkpoint predict() used (``uns[...]['ckpt']``) vs the
    ``checkpoints/*.ckpt`` listing of the run dir (last.ckpt vs best
    val_loss), i.e. what the decode's legacy 'last' policy would pick.

Nothing is written. Only numpy + anndata (and its h5py backend) are needed;
both are imported lazily. Only ``obs`` and the needed ``uns`` keys are read
(falls back to ``anndata.read_h5ad(backed='r')`` for unusual files).

Usage
-----
    # Old (transductive) run: measure the exposure, provenance and ckpt.
    python examples/check_holdout_leakage.py \\
        --pred-ref $ART/$D/$V_OLD/20260629_023326_seed0/predicted_adata.h5ad

    # New (inductive) run, checked against an old run of the same data.
    python examples/check_holdout_leakage.py \\
        --pred $ART/$D/$V_IND/<TS>_seed0/predicted_adata.h5ad \\
        --pred-ref $ART/$D/$V_OLD/20260629_023326_seed0/predicted_adata.h5ad \\
        --expect-n-test 6612

Exit status 1 when (a) --pred has any held-out -> observed edge, or its
edge_index is missing / does not index its rows, (b) --pred and --pred-ref
differ in obs_names or the held-out split, or (c) --expect-n-test does not
match a given file. Otherwise 0 (without --expect-n-test, a --pred-ref alone
only reports).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_META_KEYS = ("squint", "vqniche")          # current, legacy (pre-rename)
_PROVENANCE_KEYS = ("holdout_graph", "holdout_test_inedges", "holdout_knn_k",
                    "holdout_graph_override")


# ---------------------------------------------------------------------------
# Loading (obs + selected uns keys only)
# ---------------------------------------------------------------------------

def _import_read_elem():
    """anndata's element reader: ``anndata.io.read_elem`` (>=0.11) or
    ``anndata.experimental.read_elem`` (0.8-0.10). None if unavailable."""
    try:
        from anndata.io import read_elem  # type: ignore[attr-defined]
        return read_elem
    except Exception:  # noqa: BLE001
        pass
    try:
        from anndata.experimental import read_elem  # type: ignore[attr-defined]
        return read_elem
    except Exception:  # noqa: BLE001
        return None


def _load_light(path: Path) -> dict:
    """Return {'obs', 'edge_index', 'meta_raw', 'meta_key'} for one h5ad,
    reading only obs + uns/{edge_index,squint,vqniche} when possible."""
    read_elem = _import_read_elem()
    if read_elem is not None:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                obs = read_elem(f["obs"])
                uns = f["uns"] if "uns" in f else {}
                ei = read_elem(uns["edge_index"]) if "edge_index" in uns else None
                meta_raw, meta_key = None, None
                for k in _META_KEYS:
                    if k in uns:
                        meta_raw, meta_key = read_elem(uns[k]), k
                        break
            return {"obs": obs, "edge_index": ei,
                    "meta_raw": meta_raw, "meta_key": meta_key}
        except Exception as exc:  # noqa: BLE001
            print(f"  (element-wise read failed: {exc!r}; "
                  f"falling back to anndata.read_h5ad(backed='r'))")
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    try:
        meta_key = next((k for k in _META_KEYS if k in a.uns), None)
        return {"obs": a.obs.copy(),
                "edge_index": a.uns.get("edge_index", None),
                "meta_raw": a.uns[meta_key] if meta_key else None,
                "meta_key": meta_key}
    finally:
        try:
            a.file.close()
        except Exception:  # noqa: BLE001
            pass


def _coerce(v):
    """Undo predict()'s value stringification: 'None' -> None, '16' -> 16,
    "{'a': 1}" -> dict; paths / plain words stay strings; numpy scalars ->
    Python scalars."""
    if hasattr(v, "item") and getattr(v, "ndim", None) == 0:
        v = v.item()                                # numpy scalar (incl. np.str_)
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "None"):
            return None
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return v
    return v


def _parse_meta(raw) -> dict:
    """uns['squint'|'vqniche'] -> plain dict. Accepts a dict (values possibly
    stringified) or the whole dict stringified."""
    if raw is None:
        return {}
    if isinstance(raw, (str, bytes)):
        raw = _coerce(raw)
    if not isinstance(raw, dict):
        return {}
    return {str(k): _coerce(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def _edge_stats(np, ei, test: "np.ndarray") -> dict:
    """Edge counts by (src, dst) split membership on a (2, E) edge_index."""
    n = int(test.size)
    src = np.asarray(ei[0], dtype=np.int64)
    dst = np.asarray(ei[1], dtype=np.int64)
    if src.size and (src.min() < 0 or dst.min() < 0
                     or src.max() >= n or dst.max() >= n):
        return {"valid": False, "n_edges": int(src.size),
                "max_index": int(max(src.max(), dst.max())), "n_obs": n}
    ts, td = test[src], test[dst]
    self_loop = src == dst
    leak = ts & ~td                                     # held-out -> observed
    sl_per_node = np.bincount(src[self_loop], minlength=n)
    return {
        "valid":            True,
        "n_edges":          int(src.size),
        "n_leak":           int(leak.sum()),
        "band":             int(np.unique(dst[leak]).size),
        "ring":             int(np.unique(src[leak]).size),
        "obs_to_test":      int((~ts & td).sum()),
        "test_to_test":     int((ts & td & ~self_loop).sum()),
        "obs_to_obs":       int((~ts & ~td & ~self_loop).sum()),
        "n_self_loops":     int(self_loop.sum()),
        "nodes_no_self":    int((sl_per_node == 0).sum()),
        "nodes_multi_self": int((sl_per_node > 1).sum()),
    }


def _as_edge_index(np, ei):
    """uns['edge_index'] as an int (2, E) array, or None."""
    if ei is None:
        return None
    if hasattr(ei, "detach"):                       # torch tensor (old pickles)
        ei = ei.detach().cpu().numpy()
    ei = np.asarray(ei)
    if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
        print("  NOTE: uns['edge_index'] is (E, 2); transposing to (2, E).")
        ei = ei.T
    if ei.ndim != 2 or ei.shape[0] != 2:
        print(f"  WARNING: uns['edge_index'] has unexpected shape {ei.shape}.")
        return None
    return ei.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Checkpoint provenance
# ---------------------------------------------------------------------------

def _ckpt_report(meta: dict, pred_path: Path) -> None:
    """Stored ckpt vs the run dir's checkpoints/*.ckpt (last vs best val_loss,
    parsed from the '<epoch>-val_loss=<v>.ckpt' names like
    squint.utils.parse_test_configs.find_best_checkpoint(mode='min'))."""
    stored = meta.get("ckpt")
    run_dir = meta.get("run_dir")
    run_dir = Path(str(run_dir)) if run_dir else pred_path.parent
    print(f"  stored ckpt (predict)  : {stored}")
    print(f"  run_dir                : {run_dir}")
    ckpt_dir = None
    for cand in (run_dir / "checkpoints", run_dir / "files" / "checkpoints",
                 pred_path.parent / "checkpoints"):
        if cand.is_dir():
            ckpt_dir = cand
            break
    if ckpt_dir is None:
        print("  checkpoints dir        : <not found> (run_dir moved or not synced?)")
        return
    names = sorted(p.name for p in ckpt_dir.glob("*.ckpt"))
    print(f"  checkpoints dir        : {ckpt_dir}  ({len(names)} ckpt)")
    for nm in names:
        print(f"      {nm}")
    best, best_val = None, None
    for nm in names:
        try:
            val = float(nm.split("val_loss=")[1].split(".ckpt")[0])
        except (IndexError, ValueError):
            continue
        if best_val is None or val < best_val:
            best, best_val = nm, val
    stored_name = Path(str(stored)).name if stored else None
    has_last = "last.ckpt" in names
    print(f"  best by val_loss       : {best}")
    print(f"  stored == best         : {stored_name == best if stored_name else 'n/a'}")
    print(f"  stored file exists     : "
          f"{Path(str(stored)).is_file() if stored else 'n/a'}"
          f"{'' if not stored_name else f'  (name in listing: {stored_name in names})'}")
    if has_last and stored_name and stored_name != "last.ckpt":
        print("  NOTE: last.ckpt exists and differs from the stored ckpt, so the "
              "decode's legacy policy (--stage1-ckpt-policy last) decodes with a "
              "different checkpoint than predict() used for the codes / X_hat.")


# ---------------------------------------------------------------------------
# Per-file report
# ---------------------------------------------------------------------------

def _report_one(np, label: str, path: Path, split_key: str) -> dict:
    print("=" * 78)
    print(f"[{label}] {path}")
    print("=" * 78)
    d = _load_light(path)
    obs = d["obs"]
    if split_key not in obs.columns:
        raise SystemExit(f"[{label}] obs[{split_key!r}] missing in {path}")
    split = obs[split_key].astype(str).to_numpy()
    test = split == "test"
    names = np.asarray(obs.index.astype(str))
    n_test = int(test.sum())
    print(f"  n_obs={names.size}  n_test={n_test}  "
          f"({split_key} values: {sorted(set(split.tolist()))})")
    if "adata_batch_id" in obs.columns:
        b = obs["adata_batch_id"].astype(str).to_numpy()
        per = {bb: int((test & (b == bb)).sum()) for bb in sorted(set(b.tolist()))}
        print(f"  n_test per adata_batch_id: {per}")

    meta = _parse_meta(d["meta_raw"])
    print(f"  provenance (uns[{d['meta_key']!r}]):")
    for k in _PROVENANCE_KEYS:
        v = meta.get(k, "<absent>")
        print(f"      {k:<24s}: {v}")
    if not any(k in meta for k in _PROVENANCE_KEYS):
        print("      (no holdout_* keys: legacy / transductive predict)")
    for k in ("variant",):
        if meta.get(k):
            print(f"      {k:<24s}: {meta[k]}")

    ei = _as_edge_index(np, d["edge_index"])
    stats = None
    if ei is None:
        print("  uns['edge_index']: <missing>")
    else:
        stats = _edge_stats(np, ei, test)
        if not stats["valid"]:
            print(f"  WARNING: edge_index max index {stats['max_index']} >= n_obs "
                  f"{stats['n_obs']}: it does not index this file's rows.")
        else:
            print(f"  edges (incl. self loops): {stats['n_edges']}")
            print(f"  held-out->observed edges: {stats['n_leak']}   "
                  f"<- must be 0 for an inductive run")
            print(f"  observed boundary band  : {stats['band']} observed cells with "
                  f">=1 held-out in-neighbour")
            print(f"  held-out ring           : {stats['ring']} held-out cells with "
                  f">=1 observed out-neighbour")
            print(f"  into held-out cells     : observed->held-out {stats['obs_to_test']}, "
                  f"held-out->held-out {stats['test_to_test']} (non-self)")
            print(f"  observed->observed      : {stats['obs_to_obs']} (non-self)")
            print(f"  self loops              : {stats['n_self_loops']}  (nodes without: "
                  f"{stats['nodes_no_self']}, with >1: {stats['nodes_multi_self']})")

    print("  checkpoint:")
    _ckpt_report(meta, path)
    return {"names": names, "test": test, "n_test": n_test,
            "stats": stats, "ei": ei, "meta": meta}


def _changed_observed_neighbourhoods(np, a: dict, b: dict) -> int:
    """#observed cells whose in-neighbourhood differs between two graphs on
    the same rows (non-self edges into observed cells, compared as sets)."""
    n = int(a["test"].size)
    keys = []
    for r in (a, b):
        ei = r["ei"]
        src, dst = ei[0], ei[1]
        keep = ~r["test"][dst] & (src != dst)
        keys.append(np.unique(src[keep] * n + dst[keep]))
    diff = np.setxor1d(keys[0], keys[1], assume_unique=True)
    return int(np.unique(diff % n).size)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, default=None,
                   help="NEW (inductive) predicted_adata.h5ad; any held-out->observed "
                        "edge is a violation (exit 1).")
    p.add_argument("--pred-ref", type=Path, default=None,
                   help="Reference (e.g. old transductive) predicted_adata.h5ad; "
                        "reported only, plus identity checks against --pred.")
    p.add_argument("--split-key", type=str, default="data_split",
                   help="obs column holding the split ('test' = held out).")
    p.add_argument("--expect-n-test", type=int, default=None,
                   help="Optional expected number of held-out cells (e.g. 6612); "
                        "a mismatch in any given file is a violation.")
    args = p.parse_args(argv)
    if args.pred is None and args.pred_ref is None:
        p.error("pass --pred and/or --pred-ref")

    import numpy as np

    violations = []
    res = {}
    for label, path in (("pred", args.pred), ("pred-ref", args.pred_ref)):
        if path is None:
            continue
        if not path.is_file():
            raise SystemExit(f"[{label}] not a file: {path}")
        res[label] = _report_one(np, label, path, args.split_key)
        if args.expect_n_test is not None and res[label]["n_test"] != args.expect_n_test:
            violations.append(f"{label}: n_test={res[label]['n_test']} != "
                              f"--expect-n-test {args.expect_n_test}")

    if "pred" in res:
        st = res["pred"]["stats"]
        if st is None:
            violations.append("pred: uns['edge_index'] missing, leak-freeness unprovable")
        elif not st["valid"]:
            violations.append("pred: uns['edge_index'] does not index the obs rows")
        elif st["n_leak"] > 0:
            violations.append(f"pred: {st['n_leak']} held-out->observed edges "
                              f"({st['band']} observed cells exposed)")

    if "pred" in res and "pred-ref" in res:
        a, b = res["pred"], res["pred-ref"]
        print("=" * 78)
        print("[pred vs pred-ref]")
        print("=" * 78)
        same_names = (a["names"].size == b["names"].size
                      and bool(np.array_equal(a["names"], b["names"])))
        print(f"  obs_names identical (row for row): {same_names}")
        if not same_names:
            sa, sb = set(a["names"].tolist()), set(b["names"].tolist())
            print(f"      only in pred: {len(sa - sb)}  only in pred-ref: {len(sb - sa)}  "
                  f"same set, other order: {sa == sb}")
            violations.append("obs_names differ between pred and pred-ref")
            if sa == sb:
                pos = {nm: i for i, nm in enumerate(b["names"].tolist())}
                order = np.array([pos[nm] for nm in a["names"].tolist()])
                n_diff = int((a["test"] != b["test"][order]).sum())
                print(f"      held-out labels differing (matched by name): {n_diff}")
        else:
            n_diff = int((a["test"] != b["test"]).sum())
            print(f"  held-out split identical         : {n_diff == 0}  "
                  f"(n_test pred={a['n_test']}, pred-ref={b['n_test']}, "
                  f"rows differing={n_diff})")
            if n_diff:
                violations.append(f"{args.split_key} differs in {n_diff} rows")
            sa_, sb_ = a["stats"], b["stats"]
            if (n_diff == 0 and sa_ is not None and sb_ is not None
                    and sa_["valid"] and sb_["valid"]):
                n_chg = _changed_observed_neighbourhoods(np, a, b)
                print(f"  observed cells whose in-neighbourhood differs: {n_chg}  "
                      f"(pred-ref band: {sb_['band']}; >0 expected for "
                      f"'rebuild'/'drop', 0 for a transductive pred)")

    print("=" * 78)
    if violations:
        print(f"RESULT: VIOLATION ({len(violations)})")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("RESULT: OK" + ("" if "pred" in res else " (report only: no --pred given)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
