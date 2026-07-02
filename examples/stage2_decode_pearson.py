#!/usr/bin/env python
"""
Decode stage-2's PREDICTED codes through the frozen stage-1 SQUINT decoder and
score gene-expression Pearson on the held-out region -- the "SQUINT (imputed)"
bar for the Pearson benchmark.

Pipeline
--------
1. Load the frozen stage-1 model (VQNiche_Dual) from its checkpoint.
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
"""

from __future__ import annotations

import argparse
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
# Decode-from-codes (replicates VQNiche_Dual forward AFTER quantization).
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
    shared ``batch_embedding`` -- mirroring vqniche_dual.forward.
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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    import anndata
    import pandas as pd

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(os.path.abspath(args.predicted_adata))

    print(f"[decode] reading {args.predicted_adata}")
    adata = anndata.read_h5ad(args.predicted_adata)
    if "cell_code_indices" not in adata.obsm:
        raise KeyError("predicted_adata.obsm has no 'cell_code_indices' (true cell codes)")
    if "X_hat" not in adata.layers:
        raise KeyError("predicted_adata.layers has no 'X_hat' (needed for the self-check)")

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
    ckpt = _find_ckpt(run_dir, args.stage1_ckpt)
    print(f"[decode] loading stage-1 model from {ckpt}")
    from vqniche.models import VQNiche_Dual
    model = VQNiche_Dual.load_from_checkpoint(ckpt, map_location=device)
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

    def _decode(idx_cell_np, idx_niche_np, rows):
        """rows: global row indices into adata; returns X_hat (len(rows), G)."""
        ic = torch.as_tensor(idx_cell_np, dtype=torch.long, device=device)
        rd = torch.as_tensor(X[rows].sum(axis=1), dtype=torch.float32, device=device)
        cov = (torch.as_tensor(cov_all[rows], dtype=torch.long, device=device)
               if model.decoder_covariate_dim > 0 else None)
        with torch.no_grad():
            xhat = decode_cell_xhat(model, ic, None, cov, rd, torch)
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

    # ---- IMPUTED: decode stage-2's predicted codes ------------------------
    npz = np.load(args.stage2_codes)
    gidx = npz["global_idx"].astype(np.int64)
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
        xhat_imp = _decode(pred_cell, None, gidx)

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
            X_hat_full = X_hat_stored.copy()
            X_hat_full[gidx] = xhat_imp
            As = _spatial_knn_selfloop(np.asarray(coords_s), batch_s,
                                       args.smooth_neighs, include_self=False)
            xhat_imp = _nbr_mean(As, X_hat_full)[gidx].astype(np.float32)
            print(f"[decode] [smooth] cell prediction spatially smoothed over "
                  f"K={args.smooth_neighs} neighbors (GeST-style, self excluded)")

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
    # vqniche_dual.forward / _cache_inference_data). So for SQUINT the niche
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
            X_nbr = _nbr_mean(A, X)                            # true neighborhood mean

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
                    xhat_niche_full[obs_rows] = _decode_niche_hard(true_niche[obs_rows], obs_rows)
                xhat_niche_full[gidx] = xhat_niche_hold
                X_hat_nbr = _nbr_mean(A, xhat_niche_full)
                print(f"[decode] [nbr] NATIVE niche decoder (attribute_decoder_niche); "
                      f"mean-agg n_neighs={args.nbr_neighs} over "
                      f"{np.unique(batch).size} section(s)")
            else:
                # Aggregate the CELL prediction: X_hat_nbr = mean-agg of X_hat, the
                # SAME neighborhood smoothing the GeST / scVI / kNN bars get. Two
                # ways to land here: (a) --niche-from-cell (INTENTIONAL fair-
                # comparison ablation: smooth SQUINT's cell prediction like the
                # baselines instead of using its native niche decoder), or (b) an
                # older stage-2 run with no 'codes_niche' (automatic fallback).
                X_hat_full = X_hat_stored.copy(); X_hat_full[gidx] = xhat_imp
                X_hat_nbr = _nbr_mean(A, X_hat_full)
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

    os.makedirs(args.out_metrics_dir, exist_ok=True)
    out_csv = os.path.join(args.out_metrics_dir, "per_seed_pearson_reconstruction.csv")
    df.to_csv(out_csv, index=False)
    print(f"[decode] wrote {out_csv}")
    test_gw = df[(df.split == "test") & (df.axis == "gene_wise")
                 & (df.transform == "raw") & (df.gene_subset == "all")]
    if not test_gw.empty:
        print(f"[decode] imputed cell gene-wise raw (test): "
              f"pearson_mean = {test_gw.iloc[0]['pearson_mean']:.4f}")
    print("[decode] DONE.")


if __name__ == "__main__":
    main()
