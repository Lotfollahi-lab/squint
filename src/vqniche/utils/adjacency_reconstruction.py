from typing import Literal

import torch


def reconstruct_adjacency_matrix(
        decoder_embeddings: torch.Tensor,
        method: Literal['threshold-matmul'] = 'threshold-matmul',
        **kwargs
    ) -> torch.Tensor:
    """
    Reconstruct the graph in the form of an adjacency matrix from the decoded vectors using the specified method.
    
    Parameters
    ----------
    - decoder_embeddings: torch.Tensor
        The decoded node-wisevectors produced by the the adjacency decoder module.
        Dimensions: (num_nodes, decoder_embedding_dim)
    - method: Literal['threshold-matmul']
        The method to use to reconstruct the graph.
        Options:
            - 'threshold-matmul': Use a thresholded matrix multiplication to reconstruct the graph.
    - **kwargs: Additional keyword arguments passed to the specific reconstruction method.

    Returns
    -------
    - reconstructed_adjacency_matrix: torch.Tensor
        The reconstructed adjacency matrix.
    """
    if method == 'threshold-matmul':
        return adjacency_via_threshold_matmul(decoder_embeddings)

    else:
        raise NotImplementedError(f"Invalid method: {method}")


def adjacency_via_threshold_matmul(
        decoder_embeddings: torch.Tensor,
    ) -> torch.Tensor:
    """
    Reconstruct the adjacency matrix from the decoded node-wise vectors using a thresholded matrix multiplication.

    Parameters
    ----------
    - decoder_embeddings: torch.Tensor
        The decoded node-wise vectors produced by the the adjacency decoder module.
        Dimensions: (num_nodes, decoder_embedding_dim)

    Returns
    -------
    - adj_reconstr: torch.Tensor
        The reconstructed adjacency matrix.
        Dimensions: (num_nodes, num_nodes)
    """
    adj_reconstr = torch.matmul(decoder_embeddings.detach(), decoder_embeddings.detach().t())
    adj_reconstr = (adj_reconstr - adj_reconstr.min()) / (adj_reconstr.max() - adj_reconstr.min() + 1e-8)

    adj_reconstr[adj_reconstr < 0.5] = 0
    adj_reconstr[adj_reconstr >= 0.5] = 1

    return adj_reconstr