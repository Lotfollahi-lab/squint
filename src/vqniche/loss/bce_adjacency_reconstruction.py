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
        wt_adj_reconstr: float = 0.1
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

    Returns
    -------
    bce_adj_reconstr_loss: torch.Tensor
        The computed adjacency reconstruction loss.

    Notes
    -----
    - In VQGraph, `h_adj` is the output from the adjacency decoder module which is fed quantized node embeddings.
    - this implementation assumes that sampling and mini-batching is used in the dataloader.
    """
    # if no sampling is performed, we build and compare original and reconstructed adjacency matrices of the seed nodes in the batch with the nodes in the induced subgraph
    if edge_sampling_ratio is None:
        # next, we construct a quantized real-valued adjacency matrix from the decoded node-wise vectors
        # this is real-valued so that the model can learn potential adjacency values
        # by setting h_index_nodes=h_adj[:batch_size] and h_target_nodes=h_adj, we compute the potential adjacency for each index node of the batch with all the nodes in the induced subgraph of the batch
        # i.e. adj_quantized is a tensor of shape (batch_size, batch_size + num_sampled_nodes) where the (i, j)-th entry is the potential adjacency between the i-th node of the batch and the j-th node of the batch
        adj_quantized = construct_real_valued_adjacency_matrix(
                            h_index_nodes=h_adj[:batch_size],
                            h_target_nodes=h_adj,
                            nonlinearity=estimate_adj_kwargs['nonlinearity'],
                            k=None, # returns real-valued adjacency matrix
                        )

        # sampling-dataloaders (e.g. NeighborLoader) use local node IDs by default in the batch_edge_index. 
        # i.e. [[0, 1], [1, 2]] means that the 0-th node in the batch is connected to the 1-st node in the batch...
        # the global node IDs are the node IDs of the seed nodes in the batch with respect to the entire graph
        # e.g. the 0-th node in the batch may be the i-th node in the global graph...
        # we stay within the batch-wise scope for computing the adjacency reconstruction loss
        # this means that the loss is computing for the subgraph induced by the index nodes in the current batch
        # this is NOT equivalent to chunking the adjacency matrix into sets of rows and is thus NOT equivalent to computing the loss on the entire graph
        # finally, we subset the tensor to include adjacency information for the index nodes with all nodes in the induced subgraph
        # i.e. adj_batch is a tensor of shape (batch_size, batch_size + num_sampled_nodes) where the (i, j)-th entry is 1 if the i-th node of the batch is connected to the j-th node of the batch, and 0 otherwise
        adj_batch = edge_index_to_adjacency_tensor(
                                    edge_index=batch_edge_index,
                                )[:batch_size]

        
        # compute the positive weight for the BCE loss
        # this is the ratio of negative to positive edges in the batch
        # we use this to balance the BCE loss so that the model learns to predict the correct number of positive edges
        if use_pos_weight:
            n_pos_edges = adj_batch.sum()
            n_neg_edges = adj_batch.numel() - n_pos_edges
            pos_weight = n_neg_edges / n_pos_edges
        else:
            pos_weight = None
        
    else:
        # number of positive edges in the total number of edges in the induced subgraph of the batch
        # batch_edge_index is a tensor of shape (2, num_edges_in_batch)
        n_pos_edges = batch_edge_index.shape[1]
        
        # sample negative edges
        # batch_non_edge_index is a tensor of shape approximately (2, n_pos_edges * edge_sampling_ratio)
        # method is a memory/runtime trade-off. `sparse` will use less memory but is slower. `dense` will use more memory but is faster.
        # force_undirected is used to ensure that the negative edges are undirected
        batch_non_edge_index = negative_sampling(
            edge_index=batch_edge_index,
            num_neg_samples=int(n_pos_edges * edge_sampling_ratio),
            method='sparse',
            force_undirected=True,
        )
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
        
        # construct a quantized real-valued adjacency matrix from the decoded node-wise vectors
        adj_quantized = F.cosine_similarity(
            x1=h_adj[edge_non_edge_index[0]],
            x2=h_adj[edge_non_edge_index[1]],
            dim=1
        ).to(h_adj.device)

        # compute the positive weight for the BCE loss
        # this is the ratio of negative to positive edges in the batch
        # we use this to balance the BCE loss so that the model learns to predict the correct number of positive edges
        if use_pos_weight:
            pos_weight = torch.tensor(n_neg_edges / n_pos_edges).to(h_adj.device)
        else:
            pos_weight = None

    # compute the mean root squared error between the quantized adjacency matrix and the original adjacency matrix
    bce_adj_reconstr_loss = F.binary_cross_entropy_with_logits(
                                    input=adj_quantized,
                                    target=adj_batch.detach(),
                                    reduction='mean',
                                    pos_weight=pos_weight
                            )

    return bce_adj_reconstr_loss * wt_adj_reconstr 