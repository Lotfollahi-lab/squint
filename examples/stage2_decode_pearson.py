#!/usr/bin/env python
"""
Decode stage-2's PREDICTED codes through the frozen stage-1 SQUINT decoder and
score gene-expression Pearson on the held-out region -- the "SQUINT (imputed)"
bar for the Pearson benchmark.

Pipeline
--------
1. Load the frozen stage-1 model (Squint_Dual) from its checkpoint.
2. Read the held-out cells' PREDICTED code stacks (the npz written by
   ``holdout_split_eval`` during the stage-2 run).
3. Map codes -> z_q (sum the RVQ levels via the codebook embeddings), re-apply
   the cell->niche FiLM coupling (filmscale), concat the decoder batch
   covariate, and run the frozen NB decoder at each cell's TRUE library depth
   -> X_hat (imputed cell-level reconstruction).
4. SELF-CHECK: decode the cell's OWN (true) codes the same way and confirm it
   reproduces the stored ``layers['X_hat']`` (cell-wise Pearson ~1.0). If this
   fails, the decode path (codebook lookup / FiLM / batch-covariate mapping /
   read-depth) is misconfigured and the imputed numbers are NOT trustworthy.
5. Score Pearson on the held-out cells with the SAME formula the benchmark uses
   and write ``per_seed_pearson_reconstruction.csv`` so the figure can ingest it.

Both branches are scored:
  * CELL  branch -- X_hat[gidx] vs X[gidx] (per-held-out-cell reconstruction).
  * NICHE branch -- SQUINT's NATIVE niche decoder. Stage-2 predicts the niche
    codes too (``codes_niche`` / ``probs_niche_*`` in the npz); those are decoded
    through ``attribute_decoder_niche`` to per-cell ``xhat_niche``, and
    X_hat_nbr = 1-hop neighborhood-MEAN of xhat_niche -- exactly SQUINT's
    reconstruction-time niche definition. The per-cell prediction uses the
    model's own reconstruction (TRUE niche codes) for observed cells + the
    stage-2 imputation for held-out cells. Target = neighborhood-mean of the TRUE
    X on the same per-batch spatial kNN graph (self-loops, mean aggregation,
    ``--nbr-neighs``, matching ``_holdout_utils.compute_X_nbr``). This is the
    key asymmetry vs GeST / scVI: those have NO niche decoder, so their runners
    graph-aggregate the CELL prediction; SQUINT uses its own niche branch. (If
    an OLD npz lacks ``codes_niche``, this falls back to aggregating the cell
    prediction, with a warning.) Disable with ``--no-nbr``.
Expression is the true counts in ``predicted_adata.X``; reconstruction uses the
cell's true read depth (the task predicts COMPOSITION from spatial context, not
sequencing depth).

RMSE note: RMSE rewards the conditional mean, so a neighbor-averaging smoother
(GeST's weighted decode = softmax-weighted mean of observed profiles) has an
intrinsic squared-error edge over a single sampled discrete code path. Pass
``--decode-samples K`` (K>1) to decode K code configs sampled from the saved
per-level posteriors and AVERAGE the profiles -- a Monte-Carlo estimate of
E[profile|context], the RMSE-optimal predictor -- which closes most of that gap
while keeping the generative model. Needs ``probs_cell_*`` in the npz (run the
stage-2 eval with save_soft=True). ``--decode-mode soft`` is the cheaper (but
only approximate) deterministic analog: E[embedding] then one decode.

``--smooth-neighs K`` is a SEPARATE axis: spatially smooth the cell prediction
over each held-out cell's K nearest neighbors (self excluded), mimicking GeST
(which predicts each cell from ~30 observed neighbors). This is spatial
averaging, not posterior-sample averaging (--decode-samples) — both lower RMSE
by moving toward a local mean, but at the cost of per-cell sharpness. Cell branch
only; the native niche branch is unaffected.

Leak-free held-out protocol knobs (all opt-in; the defaults reproduce the
legacy decode numbers exactly):
  * ``--stage1-ckpt-policy {last,best,adata}``: which stage-1 checkpoint to
    decode with. 'last' (default) = legacy last.ckpt; 'best' = best val_loss
    checkpoint; 'adata' = the checkpoint predict() used to write the codes and
    X_hat (uns['squint'|'vqniche']['ckpt']), falling back to 'best'.
  * ``--observed-xhat {stored,redecode}``: observed rows of the full per-cell
    prediction come from the stored ``layers['X_hat']`` (default) or are
    re-decoded from their TRUE cell codes in observed-only chunks with the
    chosen checkpoint (no co-batching with held-out rows, no checkpoint mix).
    The CSV column records 'stored (redecode inert)' when the redecoded rows
    feed no score (native niche path without --smooth-neighs, or
    --nbr-observed-fill truth / heldout_only).
  * ``--nbr-observed-fill {stage1,truth,heldout_only}``: how observed
    neighbours enter the neighbourhood score. 'stage1' (default) = their
    stage-1 prediction; 'truth' = their true counts; 'heldout_only' = both the
    prediction and the target aggregate over held-out neighbours plus self.
  * ``--selfcheck-strict``: raise (instead of warn) when the self-check fails.
  * ``--save-pred``: also write ``heldout_predictions.npz`` next to the CSV: the
    held-out rows in adata order with the per-cell prediction exactly as the
    cell branch scored it, the true counts it was scored against, section id,
    coordinates and decode provenance (for external rescoring; join by section
    + nearest coordinate, never by obs_names).
  * ``--compare-csv REF``: reproduction gate against a reference CSV (e.g. the
    paper arm of the same seed): shared cell / niche numeric columns must agree
    within ``--compare-tol`` and, without --smooth-neighs, each x_hat row must
    sum to its read_depth. Reports the reference mtime vs commits 6b1389a /
    fdca41a (the hard No-MC decode used the TRUE held-out library size before
    fdca41a). ``--compare-strict`` raises on FAIL after writing the outputs.
An always-on guard checks that the stage-2 npz covers exactly the
``obs['data_split']=='test'`` rows of the predicted_adata.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Pearson (mirrors _holdout_utils._branch_pearson_rows / compute_inference_metrics
# EXACTLY so the number is comparable to the other benchmark bars).
# ---------------------------------------------------------------------------
def _pearson_pairwise(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_c = a - a.mean(axis=axis, keepdims=True)
    b_c = b - b.mean(axis=axis, keepdims=True)
    num = (a_c * b_c).sum(axis=axis)
    den = np.sqrt((a_c ** 2).sum(axis=axis) * (b_c ** 2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _hvg_idx(target_log1p: np.ndarray, n_hvg: int) -> np.ndarray:
    n = min(int(n_hvg), int(target_log1p.shape[1]))
    if n <= 0:
        return np.array([], dtype=int)
    gene_var = target_log1p.var(axis=0)
    idx = np.argpartition(-gene_var, n - 1)[:n]
    idx.sort()
    return idx


def _rankdata_axis(x, axis):
    from scipy.stats import rankdata
    try:
        return rankdata(x, axis=axis).astype(np.float32)
    except TypeError:                                   # scipy < 1.10: no axis=
        return np.apply_along_axis(rankdata, axis, x).astype(np.float32)


def _spearman_pairwise(a, b, axis):
    """Spearman == Pearson on average-ranks (transform-invariant)."""
    return _pearson_pairwise(_rankdata_axis(a, axis), _rankdata_axis(b, axis), axis)


def _mse_pairwise(pred, target, axis):
    return ((np.asarray(pred, np.float32) - np.asarray(target, np.float32)) ** 2).mean(axis=axis)


def _finite_mm(vec):
    v = vec[np.isfinite(vec)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(np.median(v))


def _zero_nonzero(pred_raw, target_raw):
    """Pooled AUROC / AUPRC for recovering nonzero entries on raw counts."""
    y = (np.asarray(target_raw).ravel() > 0).astype(np.int8)
    if y.size == 0 or y.min() == y.max():
        return float("nan"), float("nan")
    s = np.asarray(pred_raw, np.float32).ravel()
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or y.min() == y.max():
        return float("nan"), float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception:                                   # noqa: BLE001 (sklearn missing/degenerate)
        return float("nan"), float("nan")


def _marker_idx(target_log1p, labels, n_per_label=10, max_total=100):
    """Union of top one-vs-rest DE genes per cell type (truth-derived), numpy
    only. Mirrors _holdout_utils.build_pearson_dataframe's marker selection."""
    labels = np.asarray(labels)
    uniq = [u for u in dict.fromkeys(labels.tolist())
            if u is not None and str(u) != "nan"
            and not (isinstance(u, float) and u != u)]
    if len(uniq) < 2 or target_log1p.shape[1] == 0:
        return np.array([], dtype=int)
    npl = min(int(n_per_label), target_log1p.shape[1])
    grand = target_log1p.mean(axis=0)
    picked = set()
    for u in uniq:
        m = labels == u
        if m.sum() == 0:
            continue
        score = target_log1p[m].mean(axis=0) - grand
        picked.update(int(i) for i in np.argpartition(-score, npl - 1)[:npl])
    idx = np.array(sorted(picked), dtype=int)
    if idx.size > max_total:
        order = np.argsort(-(target_log1p[:, idx].var(axis=0)))
        idx = np.sort(idx[order[:max_total]])
    return idx


def _pearson_rows(target, pred, branch, split, seed, n_hvg=50, log1p=True,
                  marker_idx=None):
    """Full metric panel per (branch, split), identical to
    _holdout_utils.build_pearson_dataframe: each correlation row carries Pearson
    + Spearman + MSE/RMSE across gene_wise/cell_wise x log1p/raw x all/hvg/
    markers; plus entrywise zero/nonzero AUROC+AUPRC rows."""
    target = np.asarray(target, np.float32)
    pred = np.asarray(pred, np.float32)
    n_cells, n_genes = target.shape
    hvg = _hvg_idx(np.log1p(np.clip(target, 0, None)), n_hvg)
    mk = marker_idx if (marker_idx is not None and len(marker_idx) > 0) else None

    transforms = (["log1p", "raw"] if log1p else ["raw"])
    rows = []
    for transform in transforms:
        if transform == "log1p":
            t_full = np.log1p(np.clip(target, 0, None))
            p_full = np.log1p(np.clip(pred, 0, None))
        else:
            t_full, p_full = target, pred
        for axis_name, axis in (("gene_wise", 0), ("cell_wise", 1)):
            subsets = ["all"]
            if axis_name == "gene_wise":
                if hvg.size > 0:
                    subsets.append(f"hvg{hvg.size}")
                if mk is not None:
                    subsets.append("markers")
            for gene_subset in subsets:
                if gene_subset == "all":
                    t_sub, p_sub, ng = t_full, p_full, n_genes
                elif gene_subset == "markers":
                    t_sub, p_sub, ng = t_full[:, mk], p_full[:, mk], int(len(mk))
                else:
                    t_sub, p_sub, ng = t_full[:, hvg], p_full[:, hvg], int(hvg.size)
                pv = _pearson_pairwise(p_sub, t_sub, axis=axis)
                if pv[np.isfinite(pv)].size == 0:
                    continue
                pm, pmd = _finite_mm(pv)
                sm, smd = _finite_mm(_spearman_pairwise(p_sub, t_sub, axis))
                mm, mmd = _finite_mm(_mse_pairwise(p_sub, t_sub, axis))
                rows.append({
                    "seed": int(seed), "branch": branch, "split": split,
                    "axis": axis_name, "transform": transform,
                    "gene_subset": gene_subset,
                    "pearson_mean": pm, "pearson_median": pmd,
                    "spearman_mean": sm, "spearman_median": smd,
                    "mse_mean": mm, "mse_median": mmd,
                    "rmse_mean": (float(np.sqrt(mm)) if mm == mm else float("nan")),
                    "n_cells": int(n_cells), "n_genes": int(ng),
                })
    # zero/nonzero recovery (entry-wise; raw counts; transform-independent)
    zsubs = [("all", None)] + ([("markers", mk)] if mk is not None else [])
    for gs, idx in zsubs:
        t_z = target if idx is None else target[:, idx]
        p_z = pred if idx is None else pred[:, idx]
        au, ap = _zero_nonzero(p_z, t_z)
        if au != au and ap != ap:
            continue
        rows.append({
            "seed": int(seed), "branch": branch, "split": split,
            "axis": "entrywise", "transform": "counts", "gene_subset": gs,
            "auroc_zero": au, "auprc_zero": ap,
            "n_cells": int(n_cells), "n_genes": int(t_z.shape[1]),
        })
    return rows


# ---------------------------------------------------------------------------
# Decode-from-codes (replicates Squint_Dual forward AFTER quantization).
# ---------------------------------------------------------------------------
def _codebook_matrix(vq_layer):
    emb = vq_layer._codebook.embed          # (H, K, D) or (K, D)
    if emb.dim() == 3:
        emb = emb[0]                        # head 0 (num_heads=1 for these models)
    return emb                              # (K, D)


def _codes_to_zq(vq, idx, torch):
    """idx: (N, L) long tensor -> z_q (N, D) by summing the RVQ levels."""
    if hasattr(vq, "layers") and len(getattr(vq, "layers")) > 0:
        L = idx.shape[1]
        if L != len(vq.layers):
            raise ValueError(f"idx has {L} levels but VQ has {len(vq.layers)} layers")
        zq = None
        for l in range(L):
            cb = _codebook_matrix(vq.layers[l])     # (K, D)
            contrib = cb[idx[:, l]]
            zq = contrib if zq is None else zq + contrib
        return zq
    cb = _codebook_matrix(vq)
    return cb[idx[:, 0]]


def _soft_zq(vq, probs_list, torch):
    """Expected embedding from per-level SOFT distributions:
    z_q = Σ_l p_l @ codebook_l. probs_list: [(N, K_l) per RVQ level]."""
    if hasattr(vq, "layers") and len(getattr(vq, "layers")) > 0:
        zq = None
        for l, p in enumerate(probs_list):
            contrib = p @ _codebook_matrix(vq.layers[l])    # (N, D)
            zq = contrib if zq is None else zq + contrib
        return zq
    return probs_list[0] @ _codebook_matrix(vq)


def decode_cell_xhat(model, idx_cell, idx_niche, cov_idx, read_depth, torch,
                     z_q_cell=None):
    """Replicate the model's post-VQ -> attribute_decoder_cell path.

    Returns X_hat (N, n_genes) = softmax(decoder(z_q_cell [+cov])) * read_depth.
    idx_cell/idx_niche: (N, L) long; cov_idx: (N,) long or None; read_depth: (N,) float.
    If ``z_q_cell`` is given (e.g. the SOFT expected embedding), it is used
    directly and idx_cell is ignored. (The cell-branch reconstruction does not
    depend on the niche codes / FiLM, which modulate the niche branch only.)
    """
    enc = model.encoder
    if z_q_cell is None:
        z_q_cell = _codes_to_zq(enc.vq_cell, idx_cell, torch)

    if model.decoder_covariate_dim > 0:
        if cov_idx is None:
            raise ValueError("model has decoder_covariate_dim>0 but cov_idx is None")
        if getattr(model, "decoupled_decoder_covariate", False):
            cov = model.batch_embedding_cell(cov_idx)
        else:
            cov = model.batch_embedding(cov_idx)
        z_q_cell_in = torch.cat([z_q_cell, cov], dim=-1)
    else:
        z_q_cell_in = z_q_cell

    return model.attribute_decoder_cell(x=z_q_cell_in, read_depth=read_depth,
                                        conditions=None)


def decode_niche_xhat(model, idx_niche, cov_idx, read_depth, torch,
                      z_q_niche=None):
    """Replicate the model's post-VQ -> attribute_decoder_niche path (the NATIVE
    niche decoder), so SQUINT's neighborhood prediction uses its own niche
    branch rather than a graph-aggregation of the cell prediction.

    Returns per-cell xhat_niche (N, n_genes); the niche-level target compares the
    1-hop MEAN of this against the 1-hop mean of the true X (see main()). The
    filmscale cell->niche coupling is PRE-VQ (it shapes which niche code is
    picked), so it is already baked into ``idx_niche`` -- decoding from the final
    niche indices reproduces z_q_niche without re-applying FiLM (same assumption
    the ccn_mode guard in main() enforces). Niche decoder covariate uses
    ``batch_embedding_niche`` under the decoupled-covariate model, else the
    shared ``batch_embedding`` -- mirroring squint_dual.forward.
    """
    enc = model.encoder
    if z_q_niche is None:
        z_q_niche = _codes_to_zq(enc.vq_niche, idx_niche, torch)

    if model.decoder_covariate_dim > 0:
        if cov_idx is None:
            raise ValueError("model has decoder_covariate_dim>0 but cov_idx is None")
        if getattr(model, "decoupled_decoder_covariate", False):
            cov = model.batch_embedding_niche(cov_idx)
        else:
            cov = model.batch_embedding(cov_idx)
        z_q_niche_in = torch.cat([z_q_niche, cov], dim=-1)
    else:
        z_q_niche_in = z_q_niche

    return model.attribute_decoder_niche(x=z_q_niche_in, read_depth=read_depth,
                                         conditions=None)


# ---------------------------------------------------------------------------
def _find_ckpt(run_dir, explicit):
    if explicit:
        return explicit
    for sub in ("checkpoints", "files/checkpoints"):
        cands = sorted(glob.glob(os.path.join(run_dir, sub, "*.ckpt")))
        if cands:
            last = [c for c in cands if "last" in os.path.basename(c).lower()]
            return last[0] if last else cands[-1]
    raise FileNotFoundError(
        f"no .ckpt under {run_dir}/checkpoints (pass --stage1-ckpt)")


def _meta_str(v):
    """uns value -> plain str (or None). Handles bytes and size-1 arrays."""
    if v is None:
        return None
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.ravel()[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return str(v)


def _stage1_meta(adata):
    """Stage-1 provenance dict from ``adata.uns['squint']`` (or the pre-rename
    ``uns['vqniche']``), plus the key it came from. predict() writes a flat dict
    of str/int/float/bool; a whole dict stored as a string is parsed with
    ast.literal_eval. Returns ({}, None) when absent or unparsable; never raises."""
    found = []
    for key in ("squint", "vqniche"):
        try:
            v = adata.uns.get(key)
            if v is None:
                continue
            if isinstance(v, (str, bytes, np.ndarray)):
                v = ast.literal_eval(_meta_str(v))
            d = dict(v)
        except Exception:                               # noqa: BLE001 (best-effort provenance)
            continue
        if "ckpt" in d:
            return d, key
        found.append((d, key))
    return found[0] if found else ({}, None)


def _has_ckpt_dir(run_dir):
    return bool(run_dir) and any(
        os.path.isdir(os.path.join(run_dir, sub))
        for sub in ("checkpoints", os.path.join("files", "checkpoints")))


def _best_ckpt(run_dir, meta_run_dir=None):
    """Best val_loss checkpoint (what predict() auto-selects for these runs)
    via squint.utils.parse_test_configs.find_best_checkpoint. Searches the
    predicted_adata's dir, else the stage-1 run_dir recorded in uns (the
    predicted_adata may live in a separate --output-dir)."""
    from squint.utils.parse_test_configs import find_best_checkpoint
    d = run_dir
    if not _has_ckpt_dir(run_dir) and _has_ckpt_dir(meta_run_dir):
        d = meta_run_dir
    return str(find_best_checkpoint(d, mode="min", metric_name="val_loss"))


def _resolve_stored_ckpt(stored, run_dir, meta_run_dir=None):
    """Existing file for the uns-recorded ckpt path, or None. Tries the path as
    stored, then the same file name under the run dir's checkpoints/ (covers
    relative or moved paths)."""
    if not stored:
        return None
    cands = [stored]
    base = os.path.basename(stored)
    for d in (run_dir, meta_run_dir):
        if d:
            cands += [os.path.join(d, "checkpoints", base),
                      os.path.join(d, "files", "checkpoints", base)]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _same_ckpt(a, b):
    if not a or not b:
        return False
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.basename(str(a)) == os.path.basename(str(b))


def _restrict_cols(A, keep):
    """Adjacency with only the columns in boolean ``keep`` (others dropped).
    With keep = held-out mask this is the 'heldout_only' aggregation graph: a
    held-out row keeps its held-out neighbours plus self (self is a held-out
    column)."""
    import scipy.sparse as sp
    out = (A @ sp.diags(np.asarray(keep, dtype=np.float32))).tocsr()
    out.eliminate_zeros()
    return out


def _assert_holdout_coverage(adata, gidx, split_key="data_split", split_value="test"):
    """ALWAYS-ON guard: the stage-2 npz rows (global_idx) must be EXACTLY the
    held-out rows of this predicted_adata. A held-out row missing from the npz
    would keep its stage-1 own-count X_hat in the neighbourhood score (a direct
    leak); extra rows or duplicates mean the npz belongs to another adata."""
    if split_key not in adata.obs.columns:
        raise KeyError(f"[decode] coverage check: obs has no '{split_key}' column; "
                       f"cannot verify the npz covers the held-out cells")
    want = np.where(adata.obs[split_key].astype(str).to_numpy() == split_value)[0]
    got = np.unique(gidx)
    n_dup = int(gidx.size - got.size)
    if n_dup or not np.array_equal(got, want):
        n_miss = int(np.setdiff1d(want, got).size)
        n_extra = int(np.setdiff1d(got, want).size)
        raise RuntimeError(
            f"[decode] coverage check FAILED: npz global_idx has {gidx.size} rows "
            f"({n_dup} duplicate(s)) vs {want.size} rows with obs['{split_key}']=="
            f"'{split_value}': {n_miss} held-out row(s) missing from the npz, "
            f"{n_extra} npz row(s) not held out. Was the npz produced from this "
            f"predicted_adata?")
    print(f"[decode] coverage OK: npz global_idx == obs['{split_key}']=='{split_value}' "
          f"({want.size} rows)")


def _save_heldout_predictions(path, adata, gidx, xhat, read_depth, x_true, meta):
    """--save-pred: write the scored held-out per-cell prediction to ``path``.

    ``gidx`` / ``xhat`` / ``read_depth`` / ``x_true`` (the true counts the cell
    branch scored against, X[gidx]) are in the scoring (stage-2 npz) order;
    rows are re-ordered to ascending adata row (``global_idx``) and
    ``stage2_npz_row`` keeps each row's position in the scoring order. Every
    array is numeric or fixed-width unicode, so np.load needs no allow_pickle.
    ``meta`` scalars are stored as 0-d arrays (None -> ''). Written to a temp
    file first, then renamed, so a crash never leaves a truncated npz.

    To align rows with another adata (silver / baseline), join by section
    (``adata_batch_id``) plus nearest ``spatial[:, :2]`` coordinate, as
    validate_table2.sh does, and assert a 1:1 test-to-test match (6052/6052 on
    the Table 2 dataset) with max |x_true - X| == 0. Do not join by
    ``obs_names``: predicted_adata and load_silver_concat add their concat
    suffixes independently (sorted batch id vs sorted file name), so the
    suffixes only agree when those two orders coincide; use them as a
    secondary check at most."""
    gidx = np.asarray(gidx, dtype=np.int64)
    xhat = np.asarray(xhat, dtype=np.float32)
    x_true = np.asarray(x_true, dtype=np.float32)
    n_genes = int(adata.n_vars)
    for name, arr in (("x_hat", xhat), ("x_true", x_true)):
        if arr.shape != (gidx.size, n_genes):
            raise ValueError(f"[decode] --save-pred: {name} shape {arr.shape} != "
                             f"({gidx.size}, {n_genes})")
    order = np.argsort(gidx, kind="stable")
    rows = gidx[order]
    bid = np.asarray(adata.obs["adata_batch_id"].values)[rows]
    if bid.dtype.kind not in "biuf":
        bid = bid.astype(str)                       # fixed-width, no pickle
    out = {
        "global_idx": rows,
        "obs_names": np.asarray(adata.obs_names[rows], dtype=str),
        "var_names": np.asarray(adata.var_names, dtype=str),
        "spatial": np.asarray(adata.obsm["spatial"], dtype=np.float64)[rows],
        "adata_batch_id": bid,
        "x_hat": np.ascontiguousarray(xhat[order]),
        "x_true": np.ascontiguousarray(x_true[order]),
        "read_depth": np.asarray(read_depth, dtype=np.float32)[order],
        "stage2_npz_row": order.astype(np.int64),
    }
    for k, v in meta.items():
        if k in out:
            raise KeyError(f"[decode] --save-pred: metadata key '{k}' clashes with an array")
        if v is None:
            v = ""
        if isinstance(v, (bool, np.bool_)):
            out[k] = np.asarray(bool(v))
        elif isinstance(v, (int, np.integer)):
            out[k] = np.asarray(int(v), dtype=np.int64)
        elif isinstance(v, (float, np.floating)):
            out[k] = np.asarray(float(v), dtype=np.float64)
        else:
            out[k] = np.asarray(str(v))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **out)
    os.replace(tmp, path)
    return out


# --compare-csv: rows are aligned on these keys (the _pearson_rows panel).
_CMP_KEYS = ["seed", "branch", "split", "axis", "transform", "gene_subset"]
# Decode-code history the reference CSV mtime is reported against (UTC):
# 6b1389a (2026-07-02 16:07:18 +0200) added --read-depth-mode (before it every
# held-out decode used the TRUE library size); fdca41a (2026-07-02 16:46:15
# +0200) made the hard (No-MC) cell decode use the --read-depth-mode depth too
# (between the two, the hard cell decode still used the TRUE library size).
_RD_COMMITS_UTC = (("6b1389a", (2026, 7, 2, 14, 7, 18)),
                   ("fdca41a", (2026, 7, 2, 14, 46, 15)))


def _compare_metric_csvs(new_df, ref_df, tol, branches):
    """--compare-csv gate: compare every numeric column shared by ``new_df``
    and ``ref_df`` on the shared ``_CMP_KEYS`` rows of ``branches``. A value
    passes when |new - ref| <= tol * max(1, |ref|) (absolute for correlations
    and AUCs, relative for MSE / RMSE / counts); NaN only matches NaN. Fails
    when a reference row is missing from ``new_df``, a branch has no shared
    rows, or any value is off. Returns (ok, worst scaled diff, report lines)."""
    import pandas as pd
    lines = []
    miss = [k for k in _CMP_KEYS if k not in ref_df.columns]
    if miss:
        return False, float("nan"), [f"reference CSV lacks key column(s) {miss}"]
    a = new_df[new_df["branch"].astype(str).isin(branches)].copy()
    b = ref_df[ref_df["branch"].astype(str).isin(branches)].copy()
    for d in (a, b):
        for k in _CMP_KEYS:
            d[k] = d[k].astype(str)
    n_dup = int(a.duplicated(_CMP_KEYS).sum() + b.duplicated(_CMP_KEYS).sum())
    if n_dup:
        return False, float("nan"), [f"{n_dup} duplicated key row(s); cannot align"]

    def _num(s):
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    cols = [c for c in b.columns
            if c not in _CMP_KEYS and c in a.columns and _num(a[c]) and _num(b[c])]
    m = a.merge(b, on=_CMP_KEYS, how="outer", suffixes=("_new", "_ref"),
                indicator=True)
    only_new = int((m["_merge"] == "left_only").sum())
    only_ref = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    ok = bool(cols) and only_ref == 0
    lines.append(f"{len(both)} shared row(s) over {len(cols)} numeric column(s); "
                 f"{only_ref} reference row(s) missing from this run, {only_new} "
                 f"row(s) only in this run; tol={tol:g} (x max(1,|ref|))")
    extra = [c for c in a.columns if c not in b.columns and c not in _CMP_KEYS]
    if extra:
        lines.append(f"columns only in this run (not compared): {extra}")
    worst_all = 0.0
    for br in branches:
        sub = both[both["branch"] == br]
        if sub.empty:
            ok = False
            lines.append(f"branch={br}: no shared rows")
            continue
        n_bad, worst, where = 0, 0.0, ""
        for c in cols:
            x = sub[f"{c}_new"].to_numpy(np.float64)
            y = sub[f"{c}_ref"].to_numpy(np.float64)
            eq = (x == y) | (np.isnan(x) & np.isnan(y))
            with np.errstate(invalid="ignore"):
                sc_d = np.abs(x - y) / np.maximum(1.0, np.abs(y))
            bad = ~eq & ~(sc_d <= tol)                  # NaN vs number -> bad
            n_bad += int(bad.sum())
            sc_f = np.where(eq, 0.0, np.where(np.isfinite(sc_d), sc_d, np.inf))
            if sc_f.size and float(sc_f.max()) > worst:
                j = int(np.argmax(sc_f))
                worst = float(sc_f[j])
                where = (f"{c} @ {sub.iloc[j]['split']}/{sub.iloc[j]['axis']}/"
                         f"{sub.iloc[j]['transform']}/{sub.iloc[j]['gene_subset']}")
        worst_all = max(worst_all, worst)
        ok = ok and n_bad == 0
        lines.append(f"branch={br}: {len(sub)} row(s), {n_bad} value(s) off; max "
                     f"scaled |diff| {worst:.3g}" + (f" ({where})" if where else ""))
    return ok, worst_all, lines


# Observed-only chunk size for --observed-xhat redecode (= run_squint's
# predict_batch_size). The decoder MLP's layer norm normalises over the whole
# input tensor, so rows are only ever co-batched with other observed rows.
_REDECODE_CHUNK = 65536


def _cov_index(adata):
    """Decoder batch-covariate index = position of each cell's adata_batch_id
    in the sorted unique ids (matches InMemoryDatasetBlob's sorted ordering)."""
    raw = np.asarray(adata.obs["adata_batch_id"].values)
    try:
        raw_int = raw.astype(np.int64)
    except (ValueError, TypeError):
        raw_int = np.unique(raw, return_inverse=True)[1].astype(np.int64)
    uniq = np.array(sorted(np.unique(raw_int)))
    pos = {int(v): i for i, v in enumerate(uniq)}
    return np.array([pos[int(v)] for v in raw_int], dtype=np.int64), uniq.size


def _dense(x):
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


# ---------------------------------------------------------------------------
# Neighborhood (niche) branch: per-batch spatial kNN + mean aggregation, so the
# cell-level prediction can be scored at the neighborhood level against the
# neighborhood-mean of the true X. Replicates the recipe in
# _holdout_utils.spatial_knn_per_batch / compute_X_nbr (squidpy generic kNN with
# set_diag=True, mean-normalised) using sklearn so this script stays
# dependency-light; the graph is Euclidean kNN + self-loops just like squidpy's,
# so X_nbr targets are comparable to the GeST / scVI / NicheCompass niche bars.
# ---------------------------------------------------------------------------
def _spatial_knn_selfloop(coords, batch, k, include_self=True):
    """Binary CSR adjacency: each cell -> its (k nearest + self) within the same
    batch (no cross-batch edges). Row-sum = neighbor count (for mean agg).

    ``include_self=True`` (niche aggregation): self-loops kept -> k+self per row.
    ``include_self=False`` (GeST-style smoothing): self-loops stripped -> the k
    nearest OTHER cells per row (the smoothed cell never sees its own value,
    matching GeST's predict-from-observed-neighbors)."""
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors
    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]
    rows, cols = [], []
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        if idx.size == 0:
            continue
        cb = coords[idx]
        kk = int(min(k + 1, idx.size))            # self + k others (self at dist 0)
        nn = NearestNeighbors(n_neighbors=kk).fit(cb)
        _, nbr = nn.kneighbors(cb)                # (m, kk); includes self
        src = np.repeat(np.arange(idx.size), kk)
        rows.append(idx[src]); cols.append(idx[nbr.ravel()])
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(r.size, np.float32), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                               # binary (collapse any dup edge)
    if not include_self:
        A.setdiag(0.0); A.eliminate_zeros()       # drop self -> k nearest OTHERS
    return A


def _nbr_mean(A, M):
    """Neighborhood MEAN aggregation A @ M / rowcount (dense out). Matches
    _holdout_utils.compute_X_nbr(normalize='mean')."""
    num = np.asarray(A @ np.asarray(M, dtype=np.float32), dtype=np.float32)
    rs = np.asarray(A.sum(axis=1)).ravel()
    rs = np.where(rs > 0, rs, 1.0)
    return (num / rs[:, None]).astype(np.float32)


def _neighbor_read_depth(coords, batch, gidx, L, k):
    """Leak-free library size for held-out cells: for each held-out cell, the
    mean library size ``L`` of its ``k`` nearest OBSERVED (non-held-out) cells
    within the same section. Never reads a held-out cell's own counts -- this
    mirrors how the GeST / kNN baselines obtain read depth (from observed
    neighbors only), so the imputation comparison is apples-to-apples. For a
    contiguous held-out block, interior cells still get a depth from the nearest
    observed cells at/beyond the block boundary. Falls back to the per-batch
    (or global) observed mean if a section has no observed cells."""
    from sklearn.neighbors import NearestNeighbors
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    held = np.zeros(n, dtype=bool); held[gidx] = True
    L = np.asarray(L, dtype=np.float32)
    rd = np.empty(gidx.size, dtype=np.float32)
    pos = {int(g): i for i, g in enumerate(gidx)}
    global_obs_mean = float(L[~held].mean()) if (~held).any() else float(L.mean())
    for b in np.unique(batch):
        in_b = (batch == b)
        obs_b = np.where(in_b & ~held)[0]
        hold_b = np.where(in_b & held)[0]
        if hold_b.size == 0:
            continue
        if obs_b.size == 0:
            for g in hold_b:
                rd[pos[int(g)]] = global_obs_mean
            continue
        kk = int(min(k, obs_b.size))
        nn = NearestNeighbors(n_neighbors=kk).fit(coords[obs_b])
        _, nbr = nn.kneighbors(coords[hold_b])       # (|hold_b|, kk) into obs_b
        depths = L[obs_b][nbr].mean(axis=1)          # (|hold_b|,)
        for j, g in enumerate(hold_b):
            rd[pos[int(g)]] = float(depths[j])
    return rd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predicted-adata", required=True,
                   help="Stage-1 run's predicted_adata.h5ad (true X + codes + data_split).")
    p.add_argument("--stage2-codes", required=True,
                   help="stage2_predicted_codes.npz written by holdout_split_eval.")
    p.add_argument("--stage1-ckpt", default=None,
                   help="Stage-1 .ckpt (default: auto-find under the run dir).")
    p.add_argument("--out-metrics-dir", required=True,
                   help="Dir to write per_seed_pearson_reconstruction.csv into.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda/cpu (default: auto).")
    p.add_argument("--decode-mode", choices=["hard", "soft"], default="hard",
                   help="hard = argmax codes -> codebook lookup; soft = expected "
                        "embedding Σ p(c)·embed(c) from the saved per-level code "
                        "distributions (GeST-style weighted aggregation; needs the "
                        "npz to carry probs_* arrays, i.e. run_stage2 save_soft=True).")
    p.add_argument("--decode-samples", type=int, default=1,
                   help="K>1: Monte-Carlo decode -- sample K code configs from the "
                        "saved per-level posteriors (probs_cell_*), decode each, and "
                        "AVERAGE the profiles ~= E[profile|context], the RMSE-optimal "
                        "predictor. Beats one sampled config on RMSE while staying "
                        "generative. K=1 (default) uses --decode-mode. Requires "
                        "probs_cell_* in the npz (run_stage2 save_soft=True); takes "
                        "precedence over --decode-mode when K>1.")
    p.add_argument("--nbr-neighs", type=int, default=16,
                   help="Spatial kNN neighbors for the niche branch. Default 16 "
                        "(matches SQUINT's native niche graph + the imputation "
                        "baselines' new default; was 10).")
    p.add_argument("--smooth-neighs", type=int, default=0,
                   help="K>0: SPATIALLY SMOOTH the CELL-level prediction — replace "
                        "each held-out cell's decoded profile with the mean over its "
                        "K nearest spatial neighbors (self EXCLUDED), GeST-style "
                        "(GeST predicts each cell from ~30 observed neighbors). "
                        "Applied to the cell branch before scoring; the niche branch "
                        "(native niche decoder) is unaffected. 0 = off (default). "
                        "This trades sharpness for RMSE (moves toward the local mean) "
                        "— read the full panel, not just RMSE. Use 30 to match GeST.")
    p.add_argument("--nbr-batch-key", type=str, default="adata_batch_id",
                   help="obs column defining batches/sections for the per-batch "
                        "spatial graph (no cross-batch edges). Default adata_batch_id.")
    p.add_argument("--read-depth-mode", choices=["true", "neighbor"], default="true",
                   help="Library-size (read depth) used to scale the decoder for "
                        "HELD-OUT cells. 'true' = the held-out cell's own total "
                        "counts (leaks the target's depth -- valid only for the "
                        "reconstruction sanity-check). 'neighbor' = mean library "
                        "size of the K nearest OBSERVED cells (leak-free, matches "
                        "the GeST/kNN baselines); USE THIS for imputation.")
    p.add_argument("--read-depth-neighs", type=int, default=16,
                   help="K for --read-depth-mode=neighbor (nearest observed cells "
                        "whose library sizes are averaged). Default 16.")
    p.add_argument("--no-nbr", action="store_true",
                   help="Skip the neighborhood (niche) branch (cell branch only).")
    p.add_argument("--niche-from-cell", action="store_true",
                   help="Build the niche branch by GRAPH-SMOOTHING the CELL "
                        "prediction (X_hat_nbr = mean-agg of X_hat) INSTEAD of "
                        "decoding SQUINT's native niche codes. This is the EXACT "
                        "same neighborhood smoothing applied to the GeST / kNN / "
                        "scVI bars, so it puts SQUINT's niche bar on an apples-to-"
                        "apples footing with them (a fair-comparison ablation of "
                        "the native niche decoder). Default off (native decoder).")
    p.add_argument("--selfcheck-min", type=float, default=0.99,
                   help="Min cell-wise Pearson for the true-code self-check (warn below).")
    # ---- leak-free held-out protocol knobs (opt-in; defaults = legacy decode) ----
    p.add_argument("--selfcheck-strict", action="store_true",
                   help="RAISE (instead of warn) when the true-code self-check is "
                        "below --selfcheck-min.")
    p.add_argument("--stage1-ckpt-policy", choices=["last", "best", "adata"],
                   default="last",
                   help="Stage-1 checkpoint to decode with (ignored when "
                        "--stage1-ckpt is given). 'last' (default, legacy) = "
                        "last.ckpt under the predicted_adata's dir. 'best' = best "
                        "val_loss checkpoint (find_best_checkpoint, mode=min). "
                        "'adata' = the checkpoint stage-1 predict used to write the "
                        "codes and X_hat (uns['squint'] or legacy uns['vqniche'] "
                        "'ckpt'), falling back to 'best' if that file is missing. "
                        "Use 'adata' when the predicted_adata was written to a "
                        "separate --output-dir.")
    p.add_argument("--observed-xhat", choices=["stored", "redecode"], default="stored",
                   help="Observed rows of the full per-cell prediction (used by "
                        "--niche-from-cell and --smooth-neighs). 'stored' (default) "
                        "= layers['X_hat'] from stage-1 predict. 'redecode' = decode "
                        "the observed rows from their TRUE cell codes at their own "
                        "depth, in observed-only chunks, with the chosen checkpoint "
                        "(no co-batching with held-out rows, no checkpoint mix). "
                        "Prints the stored-vs-redecoded agreement. The CSV column "
                        "observed_xhat reads 'stored (redecode inert)' when the "
                        "redecoded rows feed no score.")
    p.add_argument("--nbr-observed-fill", choices=["stage1", "truth", "heldout_only"],
                   default="stage1",
                   help="How OBSERVED neighbours of held-out cells enter the "
                        "neighbourhood computations. 'stage1' (default, legacy) = "
                        "their stage-1 prediction (X_hat per --observed-xhat, or "
                        "their native niche decode). 'truth' = their TRUE counts X "
                        "(deployment-faithful; mirrored by the baseline harness). "
                        "'heldout_only' = the niche prediction AND target aggregate "
                        "over held-out neighbours plus self only (so the score "
                        "measures imputed content only). Also applied to "
                        "--smooth-neighs (heldout_only: cells left with no held-out "
                        "neighbour keep their unsmoothed prediction).")
    p.add_argument("--save-pred", action="store_true",
                   help="Also write <out-metrics-dir>/heldout_predictions.npz: "
                        "global_idx (the scored held-out rows, in adata order), "
                        "obs_names, var_names, spatial (obsm['spatial'] rows, "
                        "float64), adata_batch_id, x_hat (float32; the per-cell held-out "
                        "prediction exactly as the cell branch scored it, i.e. after "
                        "MC / soft / hard decode, --read-depth-mode and "
                        "--smooth-neighs), x_true (float32; X[global_idx], the target "
                        "it was scored against), read_depth, stage2_npz_row and scalar "
                        "decode provenance. Join to other adatas by adata_batch_id "
                        "+ nearest spatial coordinate, not by obs_names. "
                        "Off by default (no extra file).")
    p.add_argument("--compare-csv", type=str, default=None,
                   help="Reproduction gate (off by default): a reference "
                        "per_seed_pearson_reconstruction.csv (e.g. the paper arm's, "
                        "same seed and flags). After writing this run's CSV, every "
                        "shared numeric column of the --compare-branches rows is "
                        "compared; prints max scaled |diff|, rows missing on either "
                        "side, the reference mtime vs commits 6b1389a / fdca41a "
                        "(before fdca41a the hard No-MC decode used each held-out "
                        "cell's TRUE library size) and the x_hat row sum vs "
                        "read_depth check. Must not be this run's output CSV.")
    p.add_argument("--compare-tol", type=float, default=None,
                   help="--compare-csv tolerance: |new - ref| <= tol * max(1, |ref|). "
                        "Default 1e-6 for a single (hard / soft) decode, 1e-3 for "
                        "MC (--decode-samples > 1; RNG / reduction-order noise).")
    p.add_argument("--compare-branches", type=str, default="cell,niche",
                   help="Comma-separated branches --compare-csv checks (default "
                        "cell,niche; use 'cell' when this run changes the niche "
                        "branch on purpose, e.g. --nbr-observed-fill truth).")
    p.add_argument("--compare-strict", action="store_true",
                   help="With --compare-csv: raise after writing the outputs when "
                        "the gate fails (default: print FAIL only).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    import anndata
    import pandas as pd

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(os.path.abspath(args.predicted_adata))

    # --compare-csv: read the reference up front (fail fast, and before this
    # run's CSV could overwrite it).
    ref_df = ref_mtime = None
    cmp_branches = [b.strip() for b in args.compare_branches.split(",") if b.strip()]
    if args.compare_csv:
        out_csv_path = os.path.join(args.out_metrics_dir,
                                    "per_seed_pearson_reconstruction.csv")
        if os.path.realpath(args.compare_csv) == os.path.realpath(out_csv_path):
            raise ValueError("--compare-csv is the CSV this run writes; use a new "
                             "--out-metrics-dir")
        if not cmp_branches:
            raise ValueError("--compare-branches is empty")
        ref_df = pd.read_csv(args.compare_csv)
        ref_mtime = os.path.getmtime(args.compare_csv)

    print(f"[decode] reading {args.predicted_adata}")
    adata = anndata.read_h5ad(args.predicted_adata)
    if "cell_code_indices" not in adata.obsm:
        raise KeyError("predicted_adata.obsm has no 'cell_code_indices' (true cell codes)")
    if "X_hat" not in adata.layers:
        raise KeyError("predicted_adata.layers has no 'X_hat' (needed for the self-check)")
    if args.save_pred and ("spatial" not in adata.obsm
                           or "adata_batch_id" not in adata.obs.columns):
        raise KeyError("--save-pred: predicted_adata needs obsm['spatial'] and "
                       "obs['adata_batch_id'] (the npz stores them so consumers can "
                       "join by section + nearest coordinate)")

    true_cell = np.asarray(adata.obsm["cell_code_indices"]).astype(np.int64)
    # niche true codes (for the self-check FiLM path); name fallback like the reader
    niche_key = next((k for k in ("neighborhood_code_indices", "niche_code_indices")
                      if k in adata.obsm), None)
    true_niche = (np.asarray(adata.obsm[niche_key]).astype(np.int64)
                  if niche_key else np.zeros_like(true_cell))

    X = _dense(adata.X).astype(np.float32)            # true counts
    X_hat_stored = _dense(adata.layers["X_hat"]).astype(np.float32)
    cov_all, n_cov = _cov_index(adata)

    # ---- stage-1 model ----------------------------------------------------
    # Checkpoint policy. 'last' (default) and an explicit --stage1-ckpt keep the
    # legacy _find_ckpt path; 'best' / 'adata' pick the checkpoint stage-1
    # predict used to write the codes and X_hat.
    meta, meta_key = _stage1_meta(adata)
    stored_ckpt = _meta_str(meta.get("ckpt"))
    meta_run_dir = _meta_str(meta.get("run_dir"))
    ckpt_policy = "explicit" if args.stage1_ckpt else args.stage1_ckpt_policy
    if args.stage1_ckpt or args.stage1_ckpt_policy == "last":
        if args.stage1_ckpt and args.stage1_ckpt_policy != "last":
            print(f"[decode] --stage1-ckpt given; ignoring "
                  f"--stage1-ckpt-policy={args.stage1_ckpt_policy}")
        ckpt = _find_ckpt(run_dir, args.stage1_ckpt)
    elif args.stage1_ckpt_policy == "best":
        ckpt = _best_ckpt(run_dir, meta_run_dir)
    else:                                               # 'adata'
        ckpt = _resolve_stored_ckpt(stored_ckpt, run_dir, meta_run_dir)
        if ckpt is None:
            print(f"[decode] [ckpt] uns ckpt {stored_ckpt!r} not found on disk; "
                  f"falling back to the best val_loss checkpoint")
            ckpt = _best_ckpt(run_dir, meta_run_dir)
    print(f"[decode] [ckpt] policy={ckpt_policy} -> {ckpt}")
    if stored_ckpt:
        print(f"[decode] [ckpt] predict-time ckpt (uns['{meta_key}']['ckpt']) = {stored_ckpt}")
        if not _same_ckpt(ckpt, _resolve_stored_ckpt(stored_ckpt, run_dir, meta_run_dir)
                          or stored_ckpt):
            print("[decode] [ckpt] WARNING: decoding with a DIFFERENT checkpoint than "
                  "the one that wrote the codes / layers['X_hat'] (codebooks and "
                  "decoder may have drifted). Use --stage1-ckpt-policy adata.")
    print(f"[decode] loading stage-1 model from {ckpt}")
    from squint.models import Squint_Dual
    model = Squint_Dual.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    if model.decoder_covariate_dim not in (0, n_cov):
        print(f"[decode] WARNING: decoder_covariate_dim={model.decoder_covariate_dim} "
              f"!= #unique adata_batch_id={n_cov}; covariate mapping may be off "
              f"(watch the self-check).")
    ccn = getattr(model.encoder, "ccn_mode", None)
    if ccn not in (None, "film_scale", "film", "film_cont"):
        raise NotImplementedError(
            f"ccn_mode='{ccn}' alters code selection PRE-VQ and can't be replayed "
            f"from final indices. This script supports None/film_scale/film/film_cont.")
    print(f"[decode] decoder_covariate_dim={model.decoder_covariate_dim}, ccn_mode={ccn}")

    def _decode(idx_cell_np, idx_niche_np, rows, read_depth=None):
        """rows: global row indices into adata; returns X_hat (len(rows), G).
        read_depth: optional (len(rows),) tensor to scale the decoder. If None,
        uses the rows' TRUE library size -- correct for the self-check and for
        reconstruction (own counts), but a LEAK for held-out imputation, so the
        hard-decode imputation path passes the leak-free rd explicitly."""
        ic = torch.as_tensor(idx_cell_np, dtype=torch.long, device=device)
        rd_ = (read_depth if read_depth is not None
               else torch.as_tensor(X[rows].sum(axis=1), dtype=torch.float32, device=device))
        cov = (torch.as_tensor(cov_all[rows], dtype=torch.long, device=device)
               if model.decoder_covariate_dim > 0 else None)
        with torch.no_grad():
            xhat = decode_cell_xhat(model, ic, None, cov, rd_, torch)
        return xhat.detach().cpu().numpy().astype(np.float32)

    # ---- SELF-CHECK: true codes must reproduce the stored X_hat -----------
    n = X.shape[0]
    sample = np.arange(n) if n <= 20000 else np.random.default_rng(0).choice(n, 20000, replace=False)
    xhat_true = _decode(true_cell[sample], None, sample)
    cw = _pearson_pairwise(xhat_true, X_hat_stored[sample], axis=1)
    cw = cw[np.isfinite(cw)]
    sc = float(np.median(cw)) if cw.size else float("nan")
    print(f"[decode] SELF-CHECK: true-code X_hat vs stored X_hat, "
          f"cell-wise Pearson median = {sc:.4f} (expect ~1.0)")
    if not (sc >= args.selfcheck_min):
        print(f"[decode] *** WARNING: self-check {sc:.4f} < {args.selfcheck_min}. "
              f"The decode path is NOT faithfully reproducing the model -- the "
              f"imputed Pearson below is unreliable. Likely causes: batch-covariate "
              f"index mapping, FiLM coupling, or read-depth. Fix before trusting.")
        if args.selfcheck_strict:
            raise RuntimeError(f"[decode] self-check {sc:.4f} < {args.selfcheck_min} "
                               f"(--selfcheck-strict)")

    # ---- IMPUTED: decode stage-2's predicted codes ------------------------
    npz = np.load(args.stage2_codes)
    gidx = npz["global_idx"].astype(np.int64)
    _assert_holdout_coverage(adata, gidx)               # always on
    held = np.zeros(n, dtype=bool)
    held[gidx] = True
    K = max(1, int(args.decode_samples))
    # Read depth (library size) used to scale the decoder. rd_full holds a value
    # per cell: observed cells keep their own true depth; held-out cells get
    # either their true depth ('true' mode -- LEAKS, sanity-check only) or the
    # mean library size of their nearest OBSERVED cells ('neighbor' -- leak-free,
    # matches GeST/kNN). Both the cell and niche branches read from rd_full.
    L_all = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
    rd_full = L_all.copy()
    if args.read_depth_mode == "neighbor":
        coords_rd = adata.obsm.get("spatial")
        if coords_rd is None:
            print("[decode] [read-depth] obsm['spatial'] missing -- cannot use "
                  "neighbor read depth; falling back to TRUE (leaky) depth.")
        else:
            batch_rd = (np.asarray(adata.obs[args.nbr_batch_key].values)
                        if args.nbr_batch_key in adata.obs.columns
                        else np.zeros(X.shape[0], dtype=np.int64))
            rd_hold = _neighbor_read_depth(np.asarray(coords_rd), batch_rd,
                                           gidx, L_all, args.read_depth_neighs)
            rd_full[gidx] = rd_hold
            print(f"[decode] [read-depth] LEAK-FREE: held-out depth = mean library "
                  f"size of {args.read_depth_neighs} nearest OBSERVED cells; "
                  f"mean {rd_hold.mean():.1f} vs leaked-true {L_all[gidx].mean():.1f}")
    else:
        print("[decode] [read-depth] mode=true: held-out cells scaled by their OWN "
              "true library size (LEAKS target depth -- for sanity-check only, "
              "not a fair imputation setting).")
    rd = torch.as_tensor(rd_full[gidx], dtype=torch.float32, device=device)
    cov = (torch.as_tensor(cov_all[gidx], dtype=torch.long, device=device)
           if model.decoder_covariate_dim > 0 else None)

    def _need_prob_keys():
        keys = sorted(k for k in npz.files if k.startswith("probs_cell_"))
        if not keys:
            raise SystemExit(
                "need probs_cell_* in the npz (soft per-level code posteriors). "
                "Re-run the stage-2 eval with save_soft=True (run_stage2.py does "
                "this by default now).")
        return keys

    if K > 1:
        # Monte-Carlo E[profile]: sample K code configs from the per-level
        # posteriors, decode each, AVERAGE the count profiles. The mean profile
        # is the RMSE-optimal estimator (vs one sampled config). Precedence over
        # --decode-mode. Averaging in COUNT space (post read-depth scaling) is
        # what RMSE-vs-true-counts rewards.
        prob_keys = _need_prob_keys()
        print(f"[decode] {gidx.size} held-out cells; MC-average over K={K} "
              f"sampled configs ({len(prob_keys)} RVQ level(s))")
        probs = [torch.as_tensor(npz[k].astype(np.float32), device=device) for k in prob_keys]
        gen = torch.Generator(device=device)
        acc = np.zeros((gidx.size, X.shape[1]), dtype=np.float64)
        for s in range(K):
            gen.manual_seed(int(args.seed) * 100003 + s)     # reproducible per (seed, sample)
            idx = torch.cat([torch.multinomial(p, 1, generator=gen) for p in probs], dim=1)
            with torch.no_grad():
                xk = decode_cell_xhat(model, idx, None, cov, rd, torch)
            acc += xk.detach().cpu().numpy().astype(np.float64)
        xhat_imp = (acc / K).astype(np.float32)
    elif args.decode_mode == "soft":
        prob_keys = _need_prob_keys()
        print(f"[decode] {gidx.size} held-out cells; decode-mode=soft (expected embedding)")
        probs = [torch.as_tensor(npz[k].astype(np.float32), device=device) for k in prob_keys]
        with torch.no_grad():
            zq = _soft_zq(model.encoder.vq_cell, probs, torch)      # expected embedding
            xhat_imp = decode_cell_xhat(model, None, None, cov, rd, torch,
                                        z_q_cell=zq).detach().cpu().numpy().astype(np.float32)
    else:
        print(f"[decode] {gidx.size} held-out cells; decode-mode=hard (argmax codes)")
        pred_cell = npz["codes_cell"].astype(np.int64)
        xhat_imp = _decode(pred_cell, None, gidx, read_depth=rd)  # rd = leak-free (rd_full[gidx])

    # ---- Observed rows of the full per-cell prediction ---------------------
    # 'stored' (default): stage-1 layers['X_hat']. 'redecode': observed rows
    # decoded from their TRUE cell codes at their own depth, in observed-only
    # chunks (balanced, <= _REDECODE_CHUNK rows) with the chosen checkpoint.
    X_hat_obs = X_hat_stored
    if args.observed_xhat == "redecode":
        obs_rd = np.where(~held)[0]
        X_hat_obs = X_hat_stored.copy()
        if obs_rd.size:
            parts = np.array_split(obs_rd, int(np.ceil(obs_rd.size / _REDECODE_CHUNK)))
            for part in parts:
                X_hat_obs[part] = _decode(true_cell[part], None, part)
            a_rd, b_st = X_hat_obs[obs_rd], X_hat_stored[obs_rd]
            cw_rd = _pearson_pairwise(a_rd, b_st, axis=1)
            cw_rd = cw_rd[np.isfinite(cw_rd)]
            rel = (np.linalg.norm(a_rd - b_st, axis=1)
                   / np.maximum(np.linalg.norm(b_st, axis=1), 1e-12))
            print(f"[decode] [observed-xhat] redecode: {obs_rd.size} observed rows in "
                  f"{len(parts)} observed-only chunk(s); stored vs redecoded cell-wise "
                  f"Pearson median = "
                  f"{(float(np.median(cw_rd)) if cw_rd.size else float('nan')):.6f}, "
                  f"relative L2 median = {float(np.median(rel)):.3e}")

    # Provenance for the observed_xhat CSV column: True once the observed rows
    # of X_hat_obs reach a score. Only _full_prediction under --nbr-observed-fill
    # stage1 does that ('truth' uses X; 'heldout_only' drops every observed
    # column from the held-out rows' aggregation).
    xhat_obs_scored = False

    def _full_prediction(held_vals):
        """Per-cell prediction over ALL rows: held-out rows = ``held_vals``;
        observed rows = true counts X under --nbr-observed-fill truth, else the
        stage-1 X_hat source (--observed-xhat)."""
        nonlocal xhat_obs_scored
        if args.nbr_observed_fill == "stage1":
            xhat_obs_scored = True
        full = (X if args.nbr_observed_fill == "truth" else X_hat_obs).copy()
        full[gidx] = held_vals
        return full

    # ---- Optional GeST-style SPATIAL SMOOTHING of the cell prediction ------
    # Replace each held-out cell's decoded profile with the MEAN over its K
    # nearest spatial neighbors (self EXCLUDED), like GeST predicting each cell
    # from ~30 observed neighbors. Neighbors draw from the FULL prediction
    # (stage-1 recon for observed cells + imputation for held-out). Cell-branch
    # only; the native niche branch below is untouched. Trades sharpness for RMSE.
    if args.smooth_neighs and args.smooth_neighs > 0:
        coords_s = adata.obsm.get("spatial")
        if coords_s is None:
            print("[decode] [smooth] obsm['spatial'] missing -- cannot smooth; skipping.")
        else:
            batch_s = (np.asarray(adata.obs[args.nbr_batch_key].values)
                       if args.nbr_batch_key in adata.obs.columns
                       else np.zeros(adata.n_obs, dtype=np.int64))
            X_hat_full = _full_prediction(xhat_imp)
            As = _spatial_knn_selfloop(np.asarray(coords_s), batch_s,
                                       args.smooth_neighs, include_self=False)
            if args.nbr_observed_fill == "heldout_only":
                As = _restrict_cols(As, held)
                no_nbr = np.asarray(As[gidx].sum(axis=1)).ravel() == 0
            xhat_sm = _nbr_mean(As, X_hat_full)[gidx].astype(np.float32)
            if args.nbr_observed_fill == "heldout_only" and no_nbr.any():
                xhat_sm[no_nbr] = xhat_imp[no_nbr]      # no held-out neighbour: unsmoothed
                print(f"[decode] [smooth] heldout_only: {int(no_nbr.sum())} held-out "
                      f"cell(s) have no held-out neighbour; kept unsmoothed")
            xhat_imp = xhat_sm
            print(f"[decode] [smooth] cell prediction spatially smoothed over "
                  f"K={args.smooth_neighs} neighbors (GeST-style, self excluded; "
                  f"observed fill={args.nbr_observed_fill})")

    # ---- Pearson on the held-out cells ------------------------------------
    target = X[gidx]
    # marker genes: top one-vs-rest DE per cell type, derived from the TRUTH on
    # TRAIN cells (so held-out genes never inform selection) — gives the
    # 'markers' gene-subset metrics, matching build_pearson_dataframe.
    marker_idx = None
    _lab = next((k for k in ("cell_type", "cell_types", "new_annotation", "annotation")
                 if k in adata.obs.columns), None)
    if _lab is not None:
        labs = adata.obs[_lab].to_numpy()
        if "data_split" in adata.obs.columns:
            tr = adata.obs["data_split"].to_numpy() != "test"
        else:
            tr = np.ones(adata.n_obs, dtype=bool)
        if tr.sum() == 0:
            tr = np.ones(adata.n_obs, dtype=bool)
        marker_idx = _marker_idx(np.log1p(np.clip(X[tr], 0, None)), labs[tr])
        print(f"[decode] markers: {0 if marker_idx is None else len(marker_idx)} "
              f"genes from label '{_lab}' (train cells)")
    rows = (_pearson_rows(target, xhat_imp, branch="cell", split="all",
                          seed=args.seed, marker_idx=marker_idx)
            + _pearson_rows(target, xhat_imp, branch="cell", split="test",
                            seed=args.seed, marker_idx=marker_idx))

    # ---- NICHE branch: neighborhood-level (X_hat_nbr vs X_nbr) -------------
    # SQUINT has a NATIVE niche decoder (attribute_decoder_niche): niche codes ->
    # per-cell xhat_niche, and X_hat_nbr = 1-hop mean of xhat_niche (see
    # squint_dual.forward / _cache_inference_data). So for SQUINT the niche
    # prediction decodes the stage-2 PREDICTED niche codes through that decoder
    # (NOT a graph-aggregation of the cell prediction -- that fallback is only
    # for methods with no niche decoder, e.g. GeST / scVI). The full per-cell
    # xhat_niche uses the model's own reconstruction (true niche codes) for
    # observed cells + the imputation for held-out cells, then one mean
    # aggregation; target = mean-agg of the TRUE X on the same graph.
    if not args.no_nbr:
        coords = adata.obsm.get("spatial")
        has_niche_codes = "codes_niche" in npz.files
        if coords is None:
            print("[decode] [nbr] obsm['spatial'] missing -- skipping niche branch.")
        else:
            if args.nbr_batch_key in adata.obs.columns:
                batch = np.asarray(adata.obs[args.nbr_batch_key].values)
            else:
                print(f"[decode] [nbr] '{args.nbr_batch_key}' not in obs -- "
                      f"treating all cells as one section.")
                batch = np.zeros(adata.n_obs, dtype=np.int64)
            A = _spatial_knn_selfloop(np.asarray(coords), batch, args.nbr_neighs)
            # boundary exposure: share of held-out cells' aggregation weight that
            # falls on OBSERVED neighbours (what --nbr-observed-fill controls)
            A_h = A[gidx]
            w_tot = np.asarray(A_h.sum(axis=1)).ravel()
            w_obs = np.asarray(A_h @ (~held).astype(np.float32)).ravel()
            frac_obs = w_obs / np.where(w_tot > 0, w_tot, 1.0)
            print(f"[decode] [nbr] {int((w_obs > 0).sum())}/{gidx.size} held-out cells "
                  f"have >=1 observed neighbour; observed share of held-out "
                  f"aggregation weight = {100.0 * float(frac_obs.mean()):.2f}% "
                  f"(nbr-observed-fill={args.nbr_observed_fill})")
            # 'heldout_only': prediction AND target aggregate over held-out
            # neighbours plus self only (observed columns dropped from the graph)
            A_agg = _restrict_cols(A, held) if args.nbr_observed_fill == "heldout_only" else A
            X_nbr = _nbr_mean(A_agg, X)                        # true neighborhood mean

            def _cov_t(rows):
                return (torch.as_tensor(cov_all[rows], dtype=torch.long, device=device)
                        if model.decoder_covariate_dim > 0 else None)

            def _rd_t(rows):
                # rd_full: observed cells -> own depth; held-out -> leak-free
                # neighbor depth (or true, per --read-depth-mode).
                return torch.as_tensor(rd_full[rows], dtype=torch.float32, device=device)

            if has_niche_codes and not args.niche_from_cell:
                # NATIVE niche decoder for the held-out cells (same hard / soft /
                # K-sample policy as the cell branch). Aggregation is LINEAR, so
                # averaging per-cell xhat_niche over K samples then aggregating ==
                # aggregating then averaging -> exact E[nbr-mean].
                def _decode_niche_hard(idx_np, rows):
                    out = np.empty((len(rows), X.shape[1]), np.float32)
                    for st in range(0, len(rows), 20000):
                        sl = slice(st, min(st + 20000, len(rows)))
                        ic = torch.as_tensor(idx_np[sl], dtype=torch.long, device=device)
                        with torch.no_grad():
                            out[sl] = (decode_niche_xhat(model, ic, _cov_t(rows[sl]),
                                                         _rd_t(rows[sl]), torch)
                                       .detach().cpu().numpy().astype(np.float32))
                    return out

                if K > 1:
                    nkeys = sorted(k for k in npz.files if k.startswith("probs_niche_"))
                    if not nkeys:
                        raise SystemExit("--decode-samples K>1 needs probs_niche_* for the "
                                         "niche branch (re-run stage-2 eval with save_soft=True).")
                    nprobs = [torch.as_tensor(npz[k].astype(np.float32), device=device) for k in nkeys]
                    ngen = torch.Generator(device=device)
                    nacc = np.zeros((gidx.size, X.shape[1]), np.float64)
                    for s in range(K):
                        ngen.manual_seed(int(args.seed) * 100019 + s)
                        nidx = torch.cat([torch.multinomial(p, 1, generator=ngen) for p in nprobs], dim=1)
                        with torch.no_grad():
                            xk = decode_niche_xhat(model, nidx, _cov_t(gidx), _rd_t(gidx), torch)
                        nacc += xk.detach().cpu().numpy().astype(np.float64)
                    xhat_niche_hold = (nacc / K).astype(np.float32)
                elif args.decode_mode == "soft":
                    nkeys = sorted(k for k in npz.files if k.startswith("probs_niche_"))
                    if not nkeys:
                        raise SystemExit("--decode-mode soft needs probs_niche_* for the niche branch.")
                    nprobs = [torch.as_tensor(npz[k].astype(np.float32), device=device) for k in nkeys]
                    with torch.no_grad():
                        zqn = _soft_zq(model.encoder.vq_niche, nprobs, torch)
                        xhat_niche_hold = (decode_niche_xhat(model, None, _cov_t(gidx),
                                                             _rd_t(gidx), torch, z_q_niche=zqn)
                                           .detach().cpu().numpy().astype(np.float32))
                else:
                    xhat_niche_hold = _decode_niche_hard(npz["codes_niche"].astype(np.int64), gidx)

                obs_rows = np.setdiff1d(np.arange(n), gidx)
                xhat_niche_full = np.empty((n, X.shape[1]), np.float32)
                if obs_rows.size:
                    # observed rows: own native niche decode (legacy), or their
                    # TRUE counts under --nbr-observed-fill truth
                    xhat_niche_full[obs_rows] = (
                        X[obs_rows] if args.nbr_observed_fill == "truth"
                        else _decode_niche_hard(true_niche[obs_rows], obs_rows))
                xhat_niche_full[gidx] = xhat_niche_hold
                X_hat_nbr = _nbr_mean(A_agg, xhat_niche_full)
                print(f"[decode] [nbr] NATIVE niche decoder (attribute_decoder_niche); "
                      f"mean-agg n_neighs={args.nbr_neighs} over "
                      f"{np.unique(batch).size} section(s)")
                if args.observed_xhat == "redecode":
                    print("[decode] [nbr] NOTE: --observed-xhat redecode does not apply "
                          "to the native niche path (observed rows are decoded from "
                          "their niche codes); it only feeds --niche-from-cell / "
                          "--smooth-neighs.")
            else:
                # Aggregate the CELL prediction: X_hat_nbr = mean-agg of X_hat, the
                # SAME neighborhood smoothing the GeST / scVI / kNN bars get. Two
                # ways to land here: (a) --niche-from-cell (INTENTIONAL fair-
                # comparison ablation: smooth SQUINT's cell prediction like the
                # baselines instead of using its native niche decoder), or (b) an
                # older stage-2 run with no 'codes_niche' (automatic fallback).
                # Observed rows follow --observed-xhat / --nbr-observed-fill
                # (default: stored X_hat, the legacy fill).
                X_hat_full = _full_prediction(xhat_imp)
                X_hat_nbr = _nbr_mean(A_agg, X_hat_full)
                if args.niche_from_cell:
                    print("[decode] [nbr] --niche-from-cell: X_hat_nbr = mean-agg of the "
                          "CELL prediction (GeST/kNN-style smoothing), NOT the native "
                          f"niche decoder; n_neighs={args.nbr_neighs} over "
                          f"{np.unique(batch).size} section(s)")
                else:
                    print("[decode] [nbr] WARNING: no 'codes_niche' in npz -- falling back to "
                          "aggregating the CELL prediction. Re-run stage-2 for the native "
                          "niche branch.")

            rows += (_pearson_rows(X_nbr[gidx], X_hat_nbr[gidx], branch="niche",
                                   split="all", seed=args.seed, marker_idx=marker_idx)
                     + _pearson_rows(X_nbr[gidx], X_hat_nbr[gidx], branch="niche",
                                     split="test", seed=args.seed, marker_idx=marker_idx))

    df = pd.DataFrame(rows)
    # leak-free protocol provenance (legacy decode = stored / last / stage1).
    # observed_xhat records the EFFECTIVE source: a redecode that fed no score
    # leaves the numbers equal to 'stored', so it is stamped as inert.
    observed_xhat_eff = args.observed_xhat
    if args.observed_xhat == "redecode" and not xhat_obs_scored:
        observed_xhat_eff = "stored (redecode inert)"
        print(f"[decode] [observed-xhat] NOTE: redecoded observed rows fed no score "
              f"(nbr-observed-fill={args.nbr_observed_fill}; they only enter via "
              f"--niche-from-cell, the no-codes_niche fallback or --smooth-neighs "
              f"under stage1); CSV observed_xhat = '{observed_xhat_eff}'")
    df["observed_xhat"] = observed_xhat_eff
    df["ckpt_policy"] = ckpt_policy
    df["nbr_observed_fill"] = args.nbr_observed_fill

    os.makedirs(args.out_metrics_dir, exist_ok=True)
    out_csv = os.path.join(args.out_metrics_dir, "per_seed_pearson_reconstruction.csv")
    df.to_csv(out_csv, index=False)
    print(f"[decode] wrote {out_csv}")
    rel_rd = rel_true = float("nan")
    if args.save_pred or args.compare_csv:
        # Depth check on xhat_imp, the array the cell branch scored above
        # (never mutated after scoring). The decoder is softmax * depth, so
        # without --smooth-neighs each row sums to the depth it was scaled by
        # (rd_full[gidx]). Close to read_depth and away from the TRUE library
        # size = the --read-depth-mode depth was used, not the held-out truth.
        rs_h = xhat_imp.sum(axis=1, dtype=np.float64)
        rd_h = rd_full[gidx].astype(np.float64)
        L_h = L_all[gidx].astype(np.float64)
        rel_rd = float(np.max(np.abs(rs_h - rd_h) / np.maximum(rd_h, 1e-12)))
        rel_true = float(np.median(np.abs(rs_h - L_h) / np.maximum(L_h, 1e-12)))
        print(f"[decode] [depth-check] x_hat row sum vs read_depth: max rel diff "
              f"{rel_rd:.2e}; vs true library size: median rel diff {rel_true:.2e} "
              f"(read_depth={args.read_depth_mode}; expect ~0 vs read_depth unless "
              f"--smooth-neighs)")

    cmp = {}
    if args.compare_csv:
        import datetime as _dt
        tol = (args.compare_tol if args.compare_tol is not None
               else (1e-3 if K > 1 else 1e-6))
        ok_csv, worst, rep = _compare_metric_csvs(df, ref_df, tol, cmp_branches)
        ref_t = _dt.datetime.fromtimestamp(ref_mtime, tz=_dt.timezone.utc)
        print(f"[decode] [compare] reference {os.path.abspath(args.compare_csv)}")
        for ln in rep:
            print(f"[decode] [compare] {ln}")
        before = []
        for sha, t in _RD_COMMITS_UTC:
            c_t = _dt.datetime(*t, tzinfo=_dt.timezone.utc)
            if ref_t < c_t:
                before.append(sha)
            print(f"[decode] [compare] reference mtime {ref_t.isoformat()} is "
                  f"{'BEFORE' if ref_t < c_t else 'after'} commit {sha} "
                  f"({c_t.isoformat()}; indicative only, the farm checkout may "
                  f"have differed from the commit)")
        hard = K == 1 and args.decode_mode == "hard"
        if args.read_depth_mode != "true" and ("6b1389a" in before
                                               or ("fdca41a" in before and hard)):
            print("[decode] [compare] NOTE: the reference predates "
                  + ("6b1389a (every held-out decode used the TRUE library size)"
                     if "6b1389a" in before else
                     "fdca41a (its hard No-MC cell decode used each held-out "
                     "cell's TRUE library size)")
                  + ", so this run is not expected to reproduce it.")
        # depth gate: only meaningful when the cell prediction is not smoothed
        depth_ok = bool(args.smooth_neighs) or rel_rd <= 1e-4
        if not args.smooth_neighs:
            print(f"[decode] [compare] depth check {'OK' if depth_ok else 'FAILED'}: "
                  f"x_hat row sum vs read_depth max rel diff {rel_rd:.2e} (<= 1e-4)")
        ok = ok_csv and depth_ok
        print(f"[decode] [compare] {'PASS' if ok else 'FAIL'}: "
              f"branches={','.join(cmp_branches)}, tol={tol:g}, decode="
              f"{'mc' if K > 1 else args.decode_mode}, K={K}"
              + ("" if ok else "; this run does NOT reproduce the reference"))
        cmp = {"compare_csv": os.path.abspath(args.compare_csv),
               "compare_csv_mtime_utc": ref_t.isoformat(),
               "compare_branches": ",".join(cmp_branches),
               "compare_tol": tol, "compare_max_scaled_diff": worst,
               "compare_pass": ok}

    if args.save_pred:
        out_npz = os.path.join(args.out_metrics_dir, "heldout_predictions.npz")
        meta_npz = {
            "xhat_rowsum_maxrel_vs_read_depth": rel_rd,
            "xhat_rowsum_medrel_vs_true_depth": rel_true,
            "decode_mode": args.decode_mode,
            "decode_effective": ("mc" if K > 1 else args.decode_mode),
            "decode_samples": K,
            "read_depth_mode": args.read_depth_mode,
            "read_depth_neighs": args.read_depth_neighs,
            "smooth_neighs": args.smooth_neighs,
            "niche_from_cell": args.niche_from_cell,
            "nbr_neighs": args.nbr_neighs,
            "nbr_observed_fill": args.nbr_observed_fill,
            "observed_xhat": observed_xhat_eff,
            "ckpt_policy": ckpt_policy,
            "stage1_ckpt": os.path.abspath(ckpt),
            "stage1_ckpt_realpath": os.path.realpath(ckpt),
            "stage1_ckpt_predict_time": stored_ckpt,
            "predicted_adata": os.path.abspath(args.predicted_adata),
            "stage2_codes": os.path.abspath(args.stage2_codes),
            "seed": args.seed,
            "selfcheck_cellwise_pearson_median": sc,
        }
        meta_npz.update(cmp)                    # gate verdict when --compare-csv
        _save_heldout_predictions(out_npz, adata, gidx, xhat_imp, rd_full[gidx],
                                  X[gidx], meta_npz)
        print(f"[decode] wrote {out_npz} (x_hat {gidx.size}x{X.shape[1]} float32, "
              f"held-out rows in adata order; decode={'mc' if K > 1 else args.decode_mode}"
              f", K={K}, read_depth={args.read_depth_mode}, ckpt={os.path.abspath(ckpt)})")
    if cmp and not cmp["compare_pass"] and args.compare_strict:
        raise RuntimeError(f"[decode] --compare-strict: this run does not reproduce "
                           f"{cmp['compare_csv']} (outputs were written for inspection)")
    test_gw = df[(df.split == "test") & (df.axis == "gene_wise")
                 & (df.transform == "raw") & (df.gene_subset == "all")]
    if not test_gw.empty:
        print(f"[decode] imputed cell gene-wise raw (test): "
              f"pearson_mean = {test_gw.iloc[0]['pearson_mean']:.4f}")
    print("[decode] DONE.")


if __name__ == "__main__":
    main()
