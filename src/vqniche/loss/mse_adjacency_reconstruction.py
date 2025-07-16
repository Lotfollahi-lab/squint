"""
MSE adjacency reconstruction loss function.
"""

import torch
import torch.nn.functional as F

from ...utils.type_conversions import edge_index_to_adjacency_tensor
from ...utils.adjacency_reconstruction import reconstruct_adjacency_matrix as construct_real_valued_adjacency_matrix


def mse_adjacency_reconstruction(
        batch_size: int,
        h_adj: torch.Tensor,
        batch_edge_index: torch.Tensor,
        estimate_adj_kwargs: dict = {},
        wt_adj_reconstr: float = 0.1
    ) -> torch.Tensor:
    """
    Compute the mean squared error (MSE) between the estimated adjacency from the decoder module and the original adjacency.

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
    estimate_adj_kwargs: dict
        The keyword arguments for the adjacency estimation method.
    wt_adj_reconstr: float
        The scaling factor for the adjacency reconstruction loss.

    Returns
    -------
    mse_adj_reconstr_loss: torch.Tensor
        The computed adjacency reconstruction loss.

    Notes
    -----
    - In VQGraph, `h_adj` is the output from the adjacency decoder module which is fed quantized node embeddings.
    - this implementation assumes that sampling and mini-batching is used in the dataloader.
    """
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

    # compute the mean root squared error between the quantized adjacency matrix and the original adjacency matrix
    mse_adj_reconstr_loss = torch.sqrt(
                                F.mse_loss(
                                    input=adj_quantized,
                                    target=adj_batch.detach(),
                                    reduction='mean'
                                )
                            )

    return mse_adj_reconstr_loss * wt_adj_reconstr 