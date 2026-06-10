"""
Binary cross entropy (BCE) adjacency reconstruction loss function.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from vqniche.utils.type_conversions import edge_index_to_adjacency_tensor
from vqniche.utils.adjacency_reconstruction import reconstruct_adjacency_matrix as construct_real_valued_adjacency_matrix


def bce_adjacency_reconstruction_loss(
        batch_size: int,
        h_adj: torch.Tensor,
        batch_edge_index: torch.Tensor,
        edge_sampling_ratio: Optional[float] = None,
        use_pos_weight: bool = False,
        estimate_adj_kwargs: dict = {},
        wt_adj_reconstr: float = 0.1,
        node_adata_batch_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """
    Compute the binary cross entropy (BCE) between the estimated adjacency from the decoder module and the original adjacency.

    Parameters
    ----------
    batch_size: int
        The size of the batch.
    h_adj: torch.Tensor
        The output from the adjacency decoder module.
        Dimensions: (batch_size + num_sampled_nodes, decoder_embedding_dim)
    batch_edge_index: torch.Tensor
        The edge index of the batch with respect to local node IDs of seed nodes.
        Dimensions: (2, num_edges_in_batch)
    edge_sampling_ratio: Optional[float]
        The ratio of positive to negative edges to sample from the batch. For every one positive edge, we sample `edge_sampling_ratio` negative edges.
        If None, no sampling is performed and the entire adjacency matrix of the seed nodes in the batch with the nodes in the induced subgraph is used.
    use_pos_weight: bool
        Whether to use a positive weight for the BCE loss. If True, num_neg_edges / num_pos_edges is provided as weight to the BCE loss.
    estimate_adj_kwargs: dict
        The keyword arguments for the adjacency estimation method.
    wt_adj_reconstr: float
        The scaling factor for the adjacency reconstruction loss.
    node_adata_batch_ids: Optional[torch.Tensor]
        Per-node adata_batch_id (section id), shape `(B+S,)` long. When
        given, restrict the BCE to within-section pairs — cross-section
        pairs are dropped from both the candidate-negative pool and the
        positive-weight accounting. Spatial edges are intra-section by
        construction (the graph is built per-AnnData), so this only ever
        drops negatives. Default `None` keeps the legacy global behaviour.

    Returns
    -------
    bce_adj_reconstr_loss: torch.Tensor
        The computed adjacency reconstruction loss.

    Notes
    -----
    - In VQGraph, `h_adj` is the output from the adjacency decoder module which is fed quantized node embeddings.
    - this implementation assumes that sampling and mini-batching is used in the dataloader.
    """
    # IMPORTANT FIX:
    # Both branches previously produced "logits" that had already been bounded
    # by a non-linearity, while still being passed to
    # `binary_cross_entropy_with_logits` (which applies sigmoid internally).
    # That meant:
    #   - non-sampling branch: nonlinearity='sigmoid' produced values in [0,1]
    #     and BCE-with-logits sigmoid'd them again → outputs squashed into
    #     [sigmoid(0), sigmoid(1)] = [0.5, 0.73]; cannot predict 0.
    #   - sampling branch: F.cosine_similarity produced values in [-1,1] and
    #     BCE-with-logits sigmoid'd them → outputs in [0.27, 0.73]; the model
    #     could never push positive edges above 0.73 or negatives below 0.27.
    # The correct setup is: produce raw, unbounded logits (dot product) and
    # let `binary_cross_entropy_with_logits` apply the sigmoid + BCE in one
    # numerically-stable step. We ignore `estimate_adj_kwargs['nonlinearity']`
    # for the loss path; it remains relevant for the inference-time adjacency
    # reconstruction in `reconstruct_adjacency_matrix`.

    # Pre-compute per-node section ids cast to long if we'll need them.
    if node_adata_batch_ids is not None:
        section_ids = node_adata_batch_ids.long()
    else:
        section_ids = None

    # if no sampling is performed, we build and compare original and reconstructed adjacency matrices of the seed nodes in the batch with the nodes in the induced subgraph
    if edge_sampling_ratio is None:
        # Raw inner-product logits between every seed node and every node in
        # the induced subgraph. Shape: (batch_size, batch_size + num_sampled).
        adj_logits = torch.matmul(h_adj[:batch_size], h_adj.T)

        # sampling-dataloaders (e.g. NeighborLoader) use local node IDs by default in the batch_edge_index.
        # i.e. [[0, 1], [1, 2]] means that the 0-th node in the batch is connected to the 1-st node in the batch...
        # we stay within the batch-wise scope for computing the adjacency reconstruction loss
        adj_batch = edge_index_to_adjacency_tensor(
                                    edge_index=batch_edge_index,
                                )[:batch_size]

        if section_ids is None:
            # Legacy: BCE over ALL (seed, induced-subgraph-node) pairs.
            if use_pos_weight:
                n_pos_edges = adj_batch.sum()
                n_neg_edges = adj_batch.numel() - n_pos_edges
                pos_weight = n_neg_edges / n_pos_edges.clamp(min=1)
            else:
                pos_weight = None

            bce_adj_reconstr_loss = F.binary_cross_entropy_with_logits(
                                            input=adj_logits,
                                            target=adj_batch.detach().to(adj_logits.dtype),
                                            reduction='mean',
                                            pos_weight=pos_weight,
                                    )
        else:
            # Within-section: drop every cross-section pair from the BCE.
            seed_ids = section_ids[:batch_size]
            same_section = seed_ids.unsqueeze(1) == section_ids.unsqueeze(0)
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
            bce_adj_reconstr_loss = (bce_unreduced * weight).sum() / n_pairs_kept

    else:
        # number of positive edges in the total number of edges in the induced subgraph of the batch
        # batch_edge_index is a tensor of shape (2, num_edges_in_batch)
        n_pos_edges = batch_edge_index.shape[1]

        # When restricting to within-section, oversample then filter
        # cross-section negatives out. Same logic as in the cosine BCE.
        if section_ids is not None:
            n_sections = int(section_ids.max().item()) + 1 if section_ids.numel() > 0 else 1
            oversample_factor = min(max(n_sections, 2), 8)
        else:
            oversample_factor = 1

        # sample negative edges
        # batch_non_edge_index is a tensor of shape approximately (2, n_pos_edges * edge_sampling_ratio)
        batch_non_edge_index = negative_sampling(
            edge_index=batch_edge_index,
            num_neg_samples=int(n_pos_edges * edge_sampling_ratio * oversample_factor),
            method='sparse',
            force_undirected=True,
        )

        if section_ids is not None:
            src_ids = section_ids[batch_non_edge_index[0]]
            dst_ids = section_ids[batch_non_edge_index[1]]
            within = src_ids == dst_ids
            batch_non_edge_index = batch_non_edge_index[:, within]
            target_n_neg = int(n_pos_edges * edge_sampling_ratio)
            if batch_non_edge_index.shape[1] > target_n_neg:
                batch_non_edge_index = batch_non_edge_index[:, :target_n_neg]

        # this sets the exact number of sampled negative edges
        n_neg_edges = batch_non_edge_index.shape[1]

        edge_non_edge_index = torch.cat(
            [batch_edge_index, batch_non_edge_index],
            dim=1
        )

        # adj_batch is a tensor of shape (n_pos_edges + n_neg_edges)
        adj_batch = torch.cat([
            torch.ones(n_pos_edges),
            torch.zeros(n_neg_edges),
        ]).to(h_adj.device)

        # Raw inner-product logits per (sampled or true) edge. Unbounded, so
        # `binary_cross_entropy_with_logits` can drive predictions to 0 or 1.
        adj_logits = (
            h_adj[edge_non_edge_index[0]] * h_adj[edge_non_edge_index[1]]
        ).sum(dim=-1)

        # compute the positive weight for the BCE loss
        if use_pos_weight:
            pos_weight = torch.tensor(
                n_neg_edges / max(n_pos_edges, 1),
                device=h_adj.device,
                dtype=adj_logits.dtype,
            )
        else:
            pos_weight = None

        bce_adj_reconstr_loss = F.binary_cross_entropy_with_logits(
                                        input=adj_logits,
                                        target=adj_batch.detach().to(adj_logits.dtype),
                                        reduction='mean',
                                        pos_weight=pos_weight,
                                )

    return bce_adj_reconstr_loss * wt_adj_reconstr