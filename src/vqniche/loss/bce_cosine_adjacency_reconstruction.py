"""
Cosine-similarity adjacency reconstruction loss (NicheCompass-style).

The standard SQUINT `bce_adjacency_reconstruction_loss` computes raw inner-
product logits between embeddings (with or without an adjacency-decoder MLP).
This module provides a cosine-similarity variant intended for use directly
on quantized niche embeddings (`z_q_niche`) — exactly what NicheCompass uses
for its graph decoder.

Why cosine sim instead of raw inner product:
  - Removes the magnitude axis from the latent. Two cells with similar
    angular direction in latent space are pushed together regardless of
    their absolute embedding norm.
  - Makes the BCE behaviour comparable across runs / variants because the
    logit range is bounded.
  - The cosine similarity is in [-1, 1]; we scale it by 1 / temperature
    before sigmoid so the BCE can drive predictions to 0 or 1 (without the
    temperature scaling, sigmoid([-1, 1]) = [0.27, 0.73] caps the dynamic
    range).
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from vqniche.utils.type_conversions import edge_index_to_adjacency_tensor


def bce_cosine_adjacency_reconstruction_loss(
        batch_size: int,
        batch_edge_index: torch.Tensor,
        z_q_niche: Optional[torch.Tensor] = None,
        z_gnn: Optional[torch.Tensor] = None,
        edge_sampling_ratio: Optional[float] = None,
        use_pos_weight: bool = False,
        cosine_temperature: float = 0.1,
        wt_adj_reconstr: float = 1.0,
        node_adata_batch_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """
    BCE between the observed adjacency and the cosine-similarity-derived
    edge probabilities of a niche-branch embedding.

    The user picks which embedding to use via the dispatcher's
    `adj_loss_input` config flag (default `"z_gnn"`):

    - `"z_gnn"`     -> continuous post-GNN niche embedding (the new default).
                       This is what NicheCompass uses (it has no VQ at all).
                       Continuous gradients flow directly into the encoder,
                       and the codebook later inherits the spatial structure
                       when it discretises z_gnn.
    - `"z_q_niche"` -> the *quantized* niche embedding. Same formulation
                       NicheCompass would use if it had VQ, but with two
                       practical drawbacks: (a) two cells assigned the same
                       code have cos-sim = 1.0 exactly, so the BCE has no
                       further gradient signal for them; (b) STE makes the
                       gradient bypass the discrete code lookup, so the loss
                       VALUE can stay flat even while the encoder is being
                       pulled.

    Exactly one of `z_q_niche` / `z_gnn` must be supplied. The dispatcher
    picks based on `loss_kwargs['adj_loss_input']`.

    Within-section vs global pair scope
    -----------------------------------
    When `node_adata_batch_ids` is supplied (per-node section / adata_batch
    id, shape `(B+S,)`), every (seed_i, node_j) pair where
    `node_adata_batch_ids[i] != node_adata_batch_ids[j]` is dropped from
    the BCE — both as a candidate negative AND from positive-weight
    accounting. Positive edges in `batch_edge_index` are intra-section by
    construction (the spatial k-NN graph is built per-AnnData, so no
    cross-section edges can exist), so this mask only ever drops negative
    pairs.

    Rationale: two biologically-similar cells (e.g. same niche type) in
    two different sections have high cosine similarity in z_gnn / z_q_niche,
    but no graph edge between them by construction of the multi-section
    blob. The legacy global BCE treats this as a "wrong prediction" and
    pushes them apart — directly anti-integration. Restricting the loss to
    within-section pairs matches NicheCompass's slide-by-slide training
    semantics: the loss never asks "do these two cells from different
    slides look connected?" because it can't tell.

    When `node_adata_batch_ids is None` (legacy default for callers that
    don't pass it), the loss falls back to the global all-pairs behaviour.

    Other parameters
    ----------------
    batch_size: int
        Number of seed nodes in the batch (their loss is computed; their
        sampled neighbours are along for the ride).
    batch_edge_index: torch.Tensor
        Edge index of the batch in *local* node IDs (i.e. node 0 is the
        0-th seed node). Dimensions: (2, num_edges_in_batch).
    edge_sampling_ratio: Optional[float]
        If None: dense pair-wise BCE over (seed, all-in-batch) pairs.
        Else: sample `edge_sampling_ratio * num_pos_edges` negative edges
        and only compute BCE on those + the positive edges.
    use_pos_weight: bool
        If True, weight positive edges by num_neg / num_pos in the BCE so
        the loss isn't dominated by negatives (which are far more numerous
        in sparse spatial graphs).
    cosine_temperature: float
        Divisor applied to cosine similarities before sigmoid. Smaller temp
        → wider logit range → BCE can saturate predictions toward 0 or 1.
        Default 0.1 puts logits in roughly [-10, 10] for cosine in [-1, 1].
    wt_adj_reconstr: float
        Final scalar multiplier on the BCE loss. The cosine BCE has small
        absolute magnitude (~0.5–1 nat at init) — NicheCompass-style models
        typically weight it 100–500× to balance against gene-expression NB
        (~150 nats). Don't leave it at 1.0 if you also want spatial
        coherence in the codes.
    node_adata_batch_ids: Optional[torch.Tensor]
        Per-node adata_batch_id (section id), shape `(B+S,)` long. When
        given, restrict the BCE to within-section pairs (see "Within-
        section vs global pair scope" above). Pass `None` to keep legacy
        global behaviour.

    Returns
    -------
    bce_loss: torch.Tensor   scalar
    """
    # Choose the embedding: continuous z_gnn by default, quantized z_q_niche
    # if explicitly requested. Both supplied is an error.
    if z_gnn is not None and z_q_niche is not None:
        raise ValueError(
            "Pass exactly one of `z_gnn` (continuous) or `z_q_niche` "
            "(quantized) to the cosine adjacency loss; got both."
        )
    embedding = z_gnn if z_gnn is not None else z_q_niche
    if embedding is None:
        raise ValueError(
            "Cosine adjacency loss requires one of `z_gnn` or `z_q_niche`."
        )

    # L2-normalise so that x · y == cosine_similarity(x, y) for unit vectors.
    z_norm = F.normalize(embedding, p=2, dim=-1)

    # Pre-compute per-node section ids cast to long if we'll need them.
    if node_adata_batch_ids is not None:
        section_ids = node_adata_batch_ids.long()
    else:
        section_ids = None

    if edge_sampling_ratio is None:
        # ---- dense pair-wise --------------------------------------------------
        # cosine[i, j] = z_norm[i] · z_norm[j]    for i in seeds, j in induced subgraph
        cos_sim = z_norm[:batch_size] @ z_norm.T        # (batch_size, B+S)
        adj_logits = cos_sim / cosine_temperature

        adj_batch = edge_index_to_adjacency_tensor(
                        edge_index=batch_edge_index,
                    )[:batch_size]

        if section_ids is None:
            # Legacy: BCE over ALL (seed, induced-subgraph-node) pairs.
            if use_pos_weight:
                n_pos = adj_batch.sum()
                n_neg = adj_batch.numel() - n_pos
                pos_weight = n_neg / n_pos.clamp(min=1)
            else:
                pos_weight = None

            bce_loss = F.binary_cross_entropy_with_logits(
                input=adj_logits,
                target=adj_batch.detach().to(adj_logits.dtype),
                reduction='mean',
                pos_weight=pos_weight,
            )
        else:
            # Within-section: zero-weight every cross-section pair so
            # only intra-section (seed, node) pairs contribute.
            # same_section[i, j] = section_ids[seed_i] == section_ids[node_j]
            seed_ids = section_ids[:batch_size]                       # (batch_size,)
            same_section = seed_ids.unsqueeze(1) == section_ids.unsqueeze(0)
            # adj_batch positives are intra-section by construction (no
            # cross-section edges exist), so same_section already covers
            # every positive — we only ever drop cross-section negatives.
            n_pairs_kept = same_section.sum().clamp(min=1)

            if use_pos_weight:
                pos_mask = adj_batch.bool() & same_section
                n_pos = pos_mask.sum()
                n_neg = n_pairs_kept - n_pos
                pos_weight = n_neg / n_pos.clamp(min=1)
            else:
                pos_weight = None

            bce_unreduced = F.binary_cross_entropy_with_logits(
                input=adj_logits,
                target=adj_batch.detach().to(adj_logits.dtype),
                reduction='none',
                pos_weight=pos_weight,
            )
            weight = same_section.to(bce_unreduced.dtype)
            bce_loss = (bce_unreduced * weight).sum() / n_pairs_kept

    else:
        # ---- sampled positives + negatives ----------------------------------
        n_pos_edges = batch_edge_index.shape[1]

        # When restricting to within-section, oversample negatives by a
        # safety factor and then filter cross-section ones out. The
        # fraction of within-section pairs is ~1 / n_sections under
        # uniform negative sampling; multiply the request by a healthy
        # factor (max 8x) so we end up with roughly the legacy count
        # after filtering even at high section counts. The cap avoids
        # ballooning the kernel matrix when section count is huge.
        if section_ids is not None:
            n_sections = int(section_ids.max().item()) + 1 if section_ids.numel() > 0 else 1
            oversample_factor = min(max(n_sections, 2), 8)
        else:
            oversample_factor = 1

        batch_non_edge_index = negative_sampling(
            edge_index=batch_edge_index,
            num_neg_samples=int(n_pos_edges * edge_sampling_ratio * oversample_factor),
            method='sparse',
            force_undirected=True,
        )

        if section_ids is not None:
            # Drop cross-section negatives (positives are intra-section
            # by construction).
            src_ids = section_ids[batch_non_edge_index[0]]
            dst_ids = section_ids[batch_non_edge_index[1]]
            within = src_ids == dst_ids
            batch_non_edge_index = batch_non_edge_index[:, within]
            # Truncate to the originally-requested count if oversampling
            # left more than asked.
            target_n_neg = int(n_pos_edges * edge_sampling_ratio)
            if batch_non_edge_index.shape[1] > target_n_neg:
                batch_non_edge_index = batch_non_edge_index[:, :target_n_neg]

        n_neg_edges = batch_non_edge_index.shape[1]

        edge_non_edge_index = torch.cat(
            [batch_edge_index, batch_non_edge_index], dim=1
        )

        adj_batch = torch.cat([
            torch.ones(n_pos_edges),
            torch.zeros(n_neg_edges),
        ]).to(z_norm.device)

        # cos_sim per (sampled or true) edge — (z_norm[i] · z_norm[j]).
        adj_logits = (
            z_norm[edge_non_edge_index[0]] * z_norm[edge_non_edge_index[1]]
        ).sum(dim=-1) / cosine_temperature

        if use_pos_weight:
            pos_weight = torch.tensor(
                n_neg_edges / max(n_pos_edges, 1),
                device=z_norm.device,
                dtype=adj_logits.dtype,
            )
        else:
            pos_weight = None

        bce_loss = F.binary_cross_entropy_with_logits(
            input=adj_logits,
            target=adj_batch.detach().to(adj_logits.dtype),
            reduction='mean',
            pos_weight=pos_weight,
        )

    return bce_loss * wt_adj_reconstr
