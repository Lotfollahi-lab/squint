"""
MaskGIT-style iterative decoding for spatial in-painting (torch).

Given a patch with observed context cells and a held-out (masked) region, fill
the held-out cells' code stacks by parallel, confidence-scheduled decoding:

    repeat for ``steps`` iterations:
      1. encode the patch once (h depends only on observed inputs + mask)
      2. decode every masked cell's full stack coarse->fine (cell L0, L1, then
         niche L0, L1), each level conditioned on the freshly chosen coarser
         codes -- this is the RQ "depth" axis
      3. commit the highest-confidence masked cells (cosine schedule); committed
         cells become observed context for the next iteration; re-mask the rest

By the last step every cell is committed. Newly committed cells flow back into
the transformer's receptive field, so large contiguous holes are filled from the
outside in. No expression is ever read or written -- only codes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import DecodeConfig, Stage2Config
from .data import AnnDataCodeSource, inpainting_patch
from .datamodule import patch_to_arrays


def _schedule_ratio(step: int, steps: int, kind: str) -> float:
    """Fraction of the held-out cells that should REMAIN masked after ``step``."""
    r = step / steps                         # in (0, 1]
    if kind == "cosine":
        return math.cos(0.5 * math.pi * r)   # 1 -> 0
    return max(0.0, 1.0 - r)                 # linear


@torch.no_grad()
def decode_patch(
    model,
    codes: torch.Tensor,                     # (B, P, T) observed codes (masked rows ignored)
    coords: torch.Tensor,                    # (B, P, 2) normalised
    mask: torch.Tensor,                      # (B, P) bool, True == held out
    key_padding_mask: Optional[torch.Tensor] = None,
    cfg: Optional[DecodeConfig] = None,
) -> torch.Tensor:
    """Return filled code stacks (B, P, T); observed cells unchanged."""
    cfg = cfg or DecodeConfig()
    model.eval()
    device = next(model.parameters()).device
    codes = codes.to(device).clone()
    coords = coords.to(device)
    mask = mask.to(device)
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(device)
        to_fill = mask & (~key_padding_mask)
    else:
        to_fill = mask.clone()

    targets = model.targets
    Ks = [model.target_K[f"{b}__{l}"] for (b, l) in targets]

    # masked cells start "unknown" for conditioning (row K of each table).
    work = codes.clone()
    for ti, K in enumerate(Ks):
        work[:, :, ti] = torch.where(
            to_fill, torch.full_like(work[:, :, ti], K), work[:, :, ti]
        )

    n_total = to_fill.sum(dim=1)                         # (B,) held-out per patch
    steps = cfg.steps

    for step in range(1, steps + 1):
        # input mask: still-masked cells -> mask_token; committed -> their codes
        still = to_fill
        h = model.encode(work.clamp_min(0), coords, still, key_padding_mask)

        cur = work.clone()
        # stack confidence (log-prob sum) over the T levels, masked cells only
        logp = torch.zeros_like(coords[:, :, 0])         # (B, P)
        for ti, K in enumerate(Ks):
            logits = model.head_logits(h, ti, cur)       # (B, P, K)
            logp_ti = F.log_softmax(logits, dim=-1)
            if cfg.temperature and cfg.temperature > 0:
                probs = F.softmax(logits / cfg.temperature, dim=-1)
                pred = torch.multinomial(probs.reshape(-1, K), 1).reshape(logits.shape[:-1])
            else:
                pred = logits.argmax(dim=-1)
            chosen_lp = logp_ti.gather(-1, pred.unsqueeze(-1)).squeeze(-1)  # (B, P)
            cur[:, :, ti] = torch.where(still, pred, work[:, :, ti])
            logp = logp + torch.where(still, chosen_lp, torch.zeros_like(logp))

        # confidence for selection (+ annealed Gumbel noise, MaskGIT trick)
        conf = logp
        if cfg.noise_anneal > 0:
            g = -torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-9)).clamp_min(1e-9))
            conf = conf + cfg.noise_anneal * (1.0 - step / steps) * g
        conf = conf.masked_fill(~still, float("-inf"))   # only pick masked cells

        # how many should remain masked after this step
        target_remaining = torch.floor(
            _schedule_ratio(step, steps, cfg.schedule) * n_total.float()
        ).long()
        if step == steps:
            target_remaining = torch.zeros_like(target_remaining)

        B = codes.shape[0]
        for bi in range(B):
            cur_masked = int(still[bi].sum().item())
            n_reveal = cur_masked - int(target_remaining[bi].item())
            if n_reveal <= 0:
                continue
            n_reveal = min(n_reveal, cur_masked)
            top = torch.topk(conf[bi], n_reveal).indices
            work[bi, top, :] = cur[bi, top, :]
            to_fill[bi, top] = False
        # refresh "still" view for next iteration done at loop top via to_fill

    return work


@torch.no_grad()
def inpaint(
    model,
    source: AnnDataCodeSource,
    holdout_idx,
    cfg: Stage2Config,
    device: Optional[str] = None,
    observed_rows=None,
) -> Dict[str, object]:
    """In-paint a held-out region of a frozen ``predicted_adata``.

    ``observed_rows`` (optional): restrict the context to these cells only
    (e.g. the section's TRAIN cells for a true data-split holdout) so no other
    held-out cell's code leaks in as context.

    Returns a dict with:
        global_idx     : (n_holdout,) row indices that were predicted
        codes          : {branch: (n_holdout, L) int} predicted code stacks
        patch_size     : number of cells in the assembled patch
    """
    patch = inpainting_patch(
        source, holdout_idx, cfg.data,
        context_radius_mult=cfg.decode.context_radius_mult,
        observed_rows=observed_rows,
    )
    arrays = patch_to_arrays(patch, cfg.prediction_targets)
    codes = torch.from_numpy(arrays["codes"]).unsqueeze(0)        # (1, P, T)
    coords = torch.from_numpy(arrays["coords"]).unsqueeze(0)      # (1, P, 2)
    mask = torch.from_numpy(arrays["mask"]).unsqueeze(0)          # (1, P)
    if device:
        model = model.to(device)

    filled = decode_patch(model, codes, coords, mask, None, cfg.decode)[0]  # (P, T)
    filled = filled.cpu().numpy()

    # gather predicted codes for the held-out cells, per branch
    targets = cfg.prediction_targets
    hold_local = arrays["mask"]
    out_codes: Dict[str, object] = {}
    for b in cfg.branches:
        L = b.num_levels
        cols = [ti for ti, (bb, _) in enumerate(targets) if bb == b.name]
        out_codes[b.name] = filled[hold_local][:, cols].astype("int64")
    return {
        "global_idx": patch.global_idx[hold_local],
        "codes": out_codes,
        "patch_size": int(patch.size),
    }
