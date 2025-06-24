from typing import Literal

import torch


def reconstruct_adjacency_matrix(
        decoder_embeddings: torch.Tensor,
        method: Literal['threshold-matmul', 'random-edges'] = 'threshold-matmul',
        **kwargs
    ) -> torch.Tensor:
    """
    Reconstruct the graph in the form of an adjacency matrix from the decoded vectors using the specified method.
    
    Parameters
    ----------
    - decoder_embeddings: torch.Tensor
        The decoded node-wisevectors produced by the the adjacency decoder module.
        Dimensions: (num_nodes, decoder_embedding_dim)

    - method: Literal['threshold-matmul', 'random-edges']
        The method to use to reconstruct the graph.
        Options:
            - 'threshold-matmul': Use a thresholded matrix multiplication to reconstruct the graph.
            - 'random-edges': Randomly add edges between pairs of nodes.

    - **kwargs: Additional keyword arguments passed to the specific reconstruction method.

    Returns
    -------
    - reconstructed_adjacency_matrix: torch.Tensor
        The reconstructed adjacency matrix.
    """
    if method == 'threshold-matmul':
        return adjacency_via_threshold_matmul(decoder_embeddings)
    elif method == 'random-edges':
        return adjacency_via_random_edges(decoder_embeddings, **kwargs)
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


def adjacency_via_random_edges(
        decoder_embeddings: torch.Tensor,
        edge_probability: float = 0.1,
    ) -> torch.Tensor:
    """
    Reconstruct the adjacency matrix by randomly adding edges between pairs of nodes.

    Parameters
    ----------
    - decoder_embeddings: torch.Tensor
        The decoded node-wise vectors produced by the adjacency decoder module.
        Dimensions: (num_nodes, decoder_embedding_dim)
    - edge_probability: float
        The probability of adding an edge between any pair of nodes.
        Default: 0.1

    Returns
    -------
    - adj_reconstr: torch.Tensor
        The reconstructed adjacency matrix.
        Dimensions: (num_nodes, num_nodes)
    """
    num_nodes = decoder_embeddings.size(0)
    
    # Create random adjacency matrix
    adj_reconstr = torch.rand(num_nodes, num_nodes, device=decoder_embeddings.device)
    
    # Apply edge probability threshold
    adj_reconstr = (adj_reconstr < edge_probability).float()
    
    # Add self-loops (set diagonal to 1)
    adj_reconstr.fill_diagonal_(1)
    
    # Make the matrix symmetric (undirected graph)
    adj_reconstr = torch.maximum(adj_reconstr, adj_reconstr.t())
    
    return adj_reconstr
