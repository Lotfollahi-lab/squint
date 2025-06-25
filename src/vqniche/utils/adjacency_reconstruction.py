from typing import Literal

import torch


def reconstruct_adjacency_matrix(
        h_adj: torch.Tensor,
        method: Literal['threshold-matmul'] = 'threshold-matmul',
        **kwargs
    ) -> torch.Tensor:
    """
    Reconstruct the graph in the form of an adjacency matrix from the decoded vectors using the specified method.
    
    Parameters
    ----------
    - h_adj: torch.Tensor
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
        return adjacency_via_threshold_matmul(h_adj)

    else:
        raise NotImplementedError(f"Invalid method: {method}")


def adjacency_via_threshold_matmul(
        h_adj: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
    """
    Reconstruct the adjacency matrix from the decoded node-wise vectors using a thresholded matrix multiplication.

    Parameters
    ----------
    - h_adj: torch.Tensor
        The decoded node-wise vectors produced by the the adjacency decoder module.
        Dimensions: (num_nodes, decoder_embedding_dim)
    - threshold: float
        The threshold to use to reconstruct the adjacency matrix.
        Default: 0.5

    Returns
    -------
    - A_hat: torch.Tensor
        The reconstructed adjacency matrix.
        Dimensions: (num_nodes, num_nodes)
    """
    A_hat = torch.matmul(
                        h_adj,
                        h_adj.t()
                    )
    A_hat = (A_hat - A_hat.min()) / (A_hat.max() - A_hat.min() + 1e-8)

    A_hat[A_hat < threshold] = 0
    A_hat[A_hat >= threshold] = 1

    return A_hat