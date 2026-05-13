"""
Contrastive auxiliary loss on the pre-quantization cell-branch latent.

Two variants live in this module:

  - `contrastive_cell_attribute_loss`              — global (any-batch)
  - `contrastive_cell_attribute_within_batch_loss` — same-batch only

The within-batch variant restricts BOTH positive-pair selection AND
the negative-sample pool to cells sharing the same `adata_batch_id`
as the anchor. This mirrors the within-section constraint we apply to
the cosine adjacency BCE (`adj_within_section_only=True`): without
it, the NT-Xent denominator includes cross-section pairs, and the
gradient pushes biologically-similar cells from different MERFISH /
STARmap sections APART — which counters the batch-integration
objective. With it, the loss is computed independently per section
and any cross-section "this is the same cell type" signal is left
unpressured, free for the encoder / codebook to capture as a batch-
invariant cluster on its own.

Motivation
----------
The NB reconstruction loss optimises per-gene rate prediction (a
*within-cell* objective). Cell-NMI is a *between-cell* metric — it
rewards cluster separation, not per-cell reconstruction accuracy.
These objectives can diverge: a latent that perfectly reconstructs
each cell's counts may still place same-type cells in different
clusters and different-type cells in adjacent clusters.

This loss adds an explicit between-cell signal: for each seed cell,
pull together (in `z_mlp_cell` cosine-sim space) the k cells with
the most similar raw-count gene-expression profile, and push apart
the rest of the batch. The targets come from the DATA STRUCTURE
itself (raw gene expression similarity) — no labels are used, so
this is still strictly unsupervised.

This is the standard NT-Xent / SimCLR loss reformulated with
"data-structure-derived positive pairs" instead of "augmentation-
derived positive pairs". With seed batch size B and k positives per
anchor, memory is O(B²) similarity matrices — fine at B ≤ 1024.

Operates ONLY on seed cells: `quantizer_input_cell` and
`target_attr` are already sliced to `[:batch_size]` by the model
wrapper before being put into `loss_data`. Sampled neighbours are
not included in the contrastive batch.
"""

import torch
import torch.nn.functional as F


def contrastive_cell_attribute_loss(
        quantizer_input_cell: torch.Tensor,
        target_attr: torch.Tensor,
        k_pos: int = 5,
        temperature: float = 0.1,
        log_transform_gene_space: bool = True,
        wt_contrastive_cell: float = 1.0,
    ) -> torch.Tensor:
    """
    NT-Xent contrastive loss with positive pairs derived from raw-gene-
    expression similarity.

    Parameters
    ----------
    quantizer_input_cell : (B, D)
        Pre-quantization cell-branch latent (= `z_mlp_cell_path` of
        the seed cells). Gradients flow back through the cell MLP /
        shared trunk depending on encoder mode.
    target_attr : (B, n_genes)
        Raw count matrix for the same seed cells. Used to pick top-k
        gene-expression-nearest neighbours per anchor — these become
        the positive pairs.
    k_pos : int
        Number of positive pairs per anchor. 5 is a sensible default
        for ~30-50-cell-type spatial transcriptomics data: small
        enough that each positive is highly likely to be same-type,
        large enough to give a meaningful gradient.
    temperature : float
        NT-Xent softmax temperature. Standard SimCLR setting is 0.1.
        Smaller -> sharper attraction / repulsion; larger -> softer.
    log_transform_gene_space : bool
        Whether to log1p the raw counts before computing positive-
        pair similarities. Strongly recommended — raw counts have
        huge dynamic range (~0 to thousands) and a few highly-
        expressed genes dominate the cosine similarity. log1p
        stabilises this.
    wt_contrastive_cell : float
        Loss weight (added to total loss as wt * loss). Standard
        SimCLR is loss-weight 1.0; for an auxiliary objective
        alongside NB reconstruction, 0.1-1.0 is sensible.

    Returns
    -------
    loss : scalar tensor
    """
    if quantizer_input_cell.numel() == 0:
        return quantizer_input_cell.sum() * 0.0

    z = quantizer_input_cell
    x = target_attr
    B = z.shape[0]
    if B < (k_pos + 2):
        # Need at least k_pos positives + 1 anchor + 1 negative;
        # degenerate batches happen on the very last partial batch
        # of an epoch. Return 0 — Lightning sums losses across the
        # epoch so this is harmless.
        return z.sum() * 0.0

    # --- gene-expression-space similarity (no grad — target pair selection) ---
    with torch.no_grad():
        if log_transform_gene_space:
            x_for_sim = torch.log1p(torch.clamp(x, min=0.0))
        else:
            x_for_sim = x
        x_normed = F.normalize(x_for_sim, dim=-1, eps=1e-8)
        sim_x = x_normed @ x_normed.t()              # (B, B), cosine similarity
        # Drop self-similarity so top-k can't pick i itself.
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        sim_x = sim_x.masked_fill(eye, float('-inf'))
        # Indices of the k_pos most-similar OTHER cells per anchor.
        _, pos_idx = sim_x.topk(k_pos, dim=-1)        # (B, k_pos)

    # --- embedding-space similarity (with grad — the actual loss) ----
    z_normed = F.normalize(z, dim=-1, eps=1e-8)
    sim_z = z_normed @ z_normed.t() / float(temperature)  # (B, B)

    # Mask self so it doesn't enter the denominator.
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim_z = sim_z.masked_fill(eye, float('-inf'))

    # NT-Xent with multiple positives per anchor (van den Oord 2018-
    # style multi-positive variant):
    #     L_i = -log( sum_{j in P_i} exp(sim_z[i, j])
    #                 / sum_{k != i}   exp(sim_z[i, k]) )
    #         = -logsumexp(sim_z[i, P_i]) + logsumexp(sim_z[i, :])
    pos_sim = sim_z.gather(1, pos_idx)               # (B, k_pos)
    num = torch.logsumexp(pos_sim, dim=-1)           # (B,)
    denom = torch.logsumexp(sim_z, dim=-1)           # (B,)
    loss_per_anchor = -(num - denom)                 # (B,)
    loss = loss_per_anchor.mean()

    return float(wt_contrastive_cell) * loss


def contrastive_cell_attribute_within_batch_loss(
        quantizer_input_cell: torch.Tensor,
        target_attr: torch.Tensor,
        node_adata_batch_ids: torch.Tensor,
        batch_size: int,
        k_pos: int = 5,
        temperature: float = 0.1,
        log_transform_gene_space: bool = True,
        wt_contrastive_cell: float = 1.0,
    ) -> torch.Tensor:
    """
    Within-batch (= same-section) variant of the NT-Xent contrastive
    cell-attribute loss. Identical to `contrastive_cell_attribute_loss`
    except that BOTH the positive-pair candidate set AND the NT-Xent
    denominator (the negative pool) are restricted to cells with the
    same `adata_batch_id` as the anchor.

    Why
    ---
    The two AnnData sections in the canonical mmb dataset
    (MERFISH-batch82 + STARmap-batch15) have systematic per-gene
    expression differences from the underlying technology. The vanilla
    global NT-Xent denominator includes cross-section cells, so its
    gradient pushes biologically-similar cells from different sections
    APART — directly counter to the batch-integration objective that
    `+decoder-cov` and the implicit per-section spatial-graph training
    are trying to achieve. This loss leaves cross-section pairs
    unpressured (neither pulled together nor pushed apart). Any "same
    cell type across sections" alignment then comes from the cell-NB
    objective + the decoder's batch covariate, both of which are
    batch-aware by construction.

    This mirrors `adj_within_section_only=True` on the cosine
    adjacency BCE — same architectural principle, applied to the
    contrastive denominator.

    Per-anchor logic
    ----------------
    1. Compute `valid[i, j] = (batch_ids[i] == batch_ids[j]) & (i != j)`
       — j is a valid candidate for anchor i iff they share a section
       and j != i.
    2. Pick top-k_pos j by raw-gene-expression cosine similarity AMONG
       VALID j's (`sim_x` masked to -inf at non-valid positions, then
       topk). These are the positive pairs.
    3. NT-Xent denominator sums exp(sim_z[i, j] / T) over all VALID j
       (sim_z masked to -inf at non-valid positions; logsumexp handles
       -inf correctly by treating those positions as not contributing).
    4. Anchors with too few same-section cells in the mini-batch
       (`n_valid < k_pos + 1`, i.e. not even one negative beyond the
       positives) contribute 0 to the average. This is a degenerate-
       case safeguard for partial batches and rare cell types; in
       normal mini-batches (B=512, two sections ~ 256 each), every
       anchor has hundreds of valid candidates.

    Parameters
    ----------
    quantizer_input_cell : (B, D)
        Already sliced to seed cells by the model wrapper.
    target_attr : (B, n_genes)
        Already sliced to seed cells.
    node_adata_batch_ids : (N,) long
        FULL per-node batch IDs (seeds + sampled neighbours, N >= B).
        Sliced to `[:batch_size]` inside this function.
    batch_size : int
        Number of seed cells (the leading prefix of node_adata_batch_ids).
    k_pos, temperature, log_transform_gene_space, wt_contrastive_cell
        Same semantics as the global variant.

    Notes on correctness
    --------------------
    - If `node_adata_batch_ids is None`, the within-batch constraint
      can't be enforced; we raise rather than silently fall back to
      the global variant, so configuration errors surface immediately.
    - `topk` on a row with fewer than k_pos non-`-inf` entries will
      pick `-inf` entries to fill out the top-k. Those `-inf`s
      propagate into `pos_sim` and the `logsumexp` ignores them.
    - When an anchor has exactly the positives as its only same-
      section cells (n_valid == k_pos), num == denom and loss[i] = 0.
      No bug, just a degenerate signal.
    - We use `nan_to_num` as a safety net for the case n_valid == 0
      (anchor is a singleton in its section within this mini-batch);
      the `has_enough` mask then zeros that anchor's contribution.
    """
    if node_adata_batch_ids is None:
        raise ValueError(
            "contrastive_cell_attribute_within_batch_loss requires "
            "node_adata_batch_ids; either set "
            "`loss_kwargs['adj_within_section_only']=True` to keep "
            "the dispatcher populating it, or use the global "
            "contrastive_cell_attribute_loss instead."
        )
    if quantizer_input_cell.numel() == 0:
        return quantizer_input_cell.sum() * 0.0

    z = quantizer_input_cell
    x = target_attr
    B = z.shape[0]
    if B < 2:
        return z.sum() * 0.0

    # node_adata_batch_ids is the FULL set (seeds + sampled neighbours);
    # we only care about seeds since the loss operates on z_mlp_cell of
    # seed cells. Slice to the leading prefix.
    batch_ids = node_adata_batch_ids
    if batch_ids.shape[0] > B:
        batch_ids = batch_ids[:B]
    elif batch_ids.shape[0] < B:
        raise ValueError(
            f"node_adata_batch_ids has fewer entries ({batch_ids.shape[0]}) "
            f"than batch_size ({B}); cannot determine per-seed sections."
        )

    # same-batch mask, exclude self
    same_batch = batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1)        # (B, B)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    valid = same_batch & ~eye                                            # (B, B)
    n_valid = valid.sum(dim=-1)                                          # (B,)

    # --- gene-expression-space similarity (no grad — for picking positives) --
    with torch.no_grad():
        if log_transform_gene_space:
            x_for_sim = torch.log1p(torch.clamp(x, min=0.0))
        else:
            x_for_sim = x
        x_normed = F.normalize(x_for_sim, dim=-1, eps=1e-8)
        sim_x = x_normed @ x_normed.t()
        # Restrict candidate pool for top-k positives to same-section cells.
        sim_x = sim_x.masked_fill(~valid, float('-inf'))
        # topk picks the k_pos most-similar same-section cells. If a row has
        # fewer than k_pos valid candidates, the remaining picks point to
        # -inf positions; downstream logsumexp handles that correctly.
        actual_k = min(int(k_pos), B - 1)
        _, pos_idx = sim_x.topk(actual_k, dim=-1)                        # (B, actual_k)

    # --- embedding-space similarity (with grad — the actual loss) ----
    z_normed = F.normalize(z, dim=-1, eps=1e-8)
    sim_z = z_normed @ z_normed.t() / float(temperature)                  # (B, B)
    # Restrict NT-Xent denominator to same-section cells (excluding self).
    sim_z = sim_z.masked_fill(~valid, float('-inf'))

    # NT-Xent with multiple positives
    pos_sim = sim_z.gather(1, pos_idx)                                    # (B, actual_k)
    num = torch.logsumexp(pos_sim, dim=-1)                                # (B,)
    denom = torch.logsumexp(sim_z, dim=-1)                                # (B,)
    loss_per_anchor = -(num - denom)                                      # (B,)

    # Safety net: anchors with no valid same-section cell (n_valid == 0)
    # produce -inf in both num and denom; the difference is NaN. Also
    # zero anchors that don't have at least k_pos + 1 same-section cells
    # (not enough to form a non-degenerate positive vs negative split:
    # num == denom -> loss == 0 anyway, but the explicit mask avoids
    # any accidental nan contamination via gradients).
    has_enough = n_valid >= (actual_k + 1)
    loss_per_anchor = torch.nan_to_num(
        loss_per_anchor, nan=0.0, posinf=0.0, neginf=0.0,
    )
    loss_per_anchor = torch.where(
        has_enough, loss_per_anchor, torch.zeros_like(loss_per_anchor),
    )
    n_active = has_enough.sum().clamp(min=1)
    loss = loss_per_anchor.sum() / n_active

    return float(wt_contrastive_cell) * loss
