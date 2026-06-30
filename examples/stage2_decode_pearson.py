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

This is the CELL branch only (the niche branch needs neighborhood aggregation of
the decoder output -- a follow-up). Expression is the true counts in
``predicted_adata.X``; reconstruction uses the cell's true read depth (the task
predicts COMPOSITION from spatial context, not sequencing depth).
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


def _pearson_rows(target, pred, branch, split, seed, n_hvg=50, log1p=True):
    """Replicates build_pearson_dataframe's per-(branch,split) rows."""
    target = np.asarray(target, np.float32)
    pred = np.asarray(pred, np.float32)
    n_cells, n_genes = target.shape
    hvg = _hvg_idx(np.log1p(np.clip(target, 0, None)), n_hvg)

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
            if axis_name == "gene_wise" and hvg.size > 0:
                subsets.append(f"hvg{hvg.size}")
            for gene_subset in subsets:
                if gene_subset == "all":
                    t_sub, p_sub, ng = t_full, p_full, n_genes
                else:
                    t_sub, p_sub, ng = t_full[:, hvg], p_full[:, hvg], int(hvg.size)
                vec = _pearson_pairwise(p_sub, t_sub, axis=axis)
                vec = vec[np.isfinite(vec)]
                if vec.size == 0:
                    continue
                rows.append({
                    "seed": int(seed), "branch": branch, "split": split,
                    "axis": axis_name, "transform": transform,
                    "gene_subset": gene_subset,
                    "pearson_mean": float(vec.mean()),
                    "pearson_median": float(np.median(vec)),
                    "n_cells": int(n_cells), "n_genes": int(ng),
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
    print(f"[decode] {gidx.size} held-out cells; decode-mode={args.decode_mode}")
    if args.decode_mode == "soft":
        prob_keys = sorted(k for k in npz.files if k.startswith("probs_cell_"))
        if not prob_keys:
            raise SystemExit(
                "--decode-mode soft needs probs_cell_* in the npz. Re-run the "
                "stage-2 eval with save_soft=True (run_stage2.py does this by "
                "default now) so soft code distributions are persisted.")
        probs = [torch.as_tensor(npz[k].astype(np.float32), device=device) for k in prob_keys]
        rd = torch.as_tensor(X[gidx].sum(axis=1), dtype=torch.float32, device=device)
        cov = (torch.as_tensor(cov_all[gidx], dtype=torch.long, device=device)
               if model.decoder_covariate_dim > 0 else None)
        with torch.no_grad():
            zq = _soft_zq(model.encoder.vq_cell, probs, torch)      # expected embedding
            xhat_imp = decode_cell_xhat(model, None, None, cov, rd, torch,
                                        z_q_cell=zq).detach().cpu().numpy().astype(np.float32)
    else:
        pred_cell = npz["codes_cell"].astype(np.int64)
        xhat_imp = _decode(pred_cell, None, gidx)

    # ---- Pearson on the held-out cells ------------------------------------
    target = X[gidx]
    rows = (_pearson_rows(target, xhat_imp, branch="cell", split="all", seed=args.seed)
            + _pearson_rows(target, xhat_imp, branch="cell", split="test", seed=args.seed))
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
