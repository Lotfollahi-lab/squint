from typing import Literal, Optional

import torch


def reconstruct_adjacency_matrix(
        h_index_nodes: torch.Tensor,
        h_target_nodes: Optional[torch.Tensor] = None,
        nonlinearity: Literal['min-max', 'sigmoid', 'softmax', 'relu-clamp'] = 'relu-clamp',
        k: Optional[int] = None,
    ) -> torch.Tensor:
    """
    Reconstruct the adjacency matrix from the decoded node-wise vectors using a thresholded matrix multiplication.

    Parameters
    ----------
    - h_index_nodes: torch.Tensor
        The decoded node-wise vectors produced by the the adjacency decoder module for the index (e.g. batch) nodes.
        Dimensions: (num_index_nodes, decoder_embedding_dim)
    - h_target_nodes: Optional[torch.Tensor]
        The decoded node-wise vectors produced by the the adjacency decoder module for the target (e.g. sampled) nodes. If None, the target nodes are assumed to be the same as the index nodes.
        Dimensions: (num_target_nodes, decoder_embedding_dim).
        Default: None
    - nonlinearity: Literal['min-max', 'sigmoid', 'softmax', 'relu-clamp']
        The nonlinearity function to apply to the matrix product.
        - 'min-max': Min-max normalization to [0, 1] range
        - 'sigmoid': Sigmoid function for probability interpretation
        - 'softmax': Row-wise softmax for probability distributions
        - 'relu-clamp': ReLU followed by clamping to [0, 1]
        Default: 'min-max'
    - k: Optional[int]
        The number of top-k values to use to reconstruct a binary adjacency matrix.
        Default: 8

    Returns
    -------
    - A_hat: torch.Tensor
        The reconstructed adjacency matrix.
        Dimensions: (num_nodes, num_nodes)
        
    Notes
    -----
    - If k is provided, it is used to convert the real-valued adjacency matrix to a binary adjacency matrix.
    - If k is not provided, the real-valued adjacency matrix is returned.
    """
    # construct a real-valued adjacency matrix from the decoded node-wise vectors
    if h_target_nodes is None:
        # if target nodes are not provided, use index nodes for both sides
        A_hat = torch.matmul(h_index_nodes, h_index_nodes.T)
    else:
        # if target nodes are provided, multiply index nodes with target nodes
        A_hat = torch.matmul(h_index_nodes, h_target_nodes.T)
    
    # apply the specified nonlinearity
    if nonlinearity == 'min-max':
        # min-max normalization to [0, 1] range
        A_hat = (A_hat - A_hat.min()) / (A_hat.max() - A_hat.min() + 1e-8)
    
    elif nonlinearity == 'sigmoid':
        # sigmoid function for probability interpretation
        A_hat = torch.sigmoid(A_hat)
    
    elif nonlinearity == 'softmax':
        # row-wise softmax for probability distributions
        A_hat = torch.softmax(A_hat, dim=1)
    
    elif nonlinearity == 'relu-clamp':
        # relu followed by clamping to [0, 1]
        A_hat = torch.relu(A_hat)
        A_hat = torch.clamp(A_hat, 0, 1)
        
    # if k is provided, use top-k to convert to binary symmetric adjacency matrix
    if k is not None:
        # get top-k indices for each row
        topk_indices = torch.topk(A_hat, k=k, dim=1)[1]

        # create a zero tensor of the same shape as A_hat
        A_hat = torch.zeros_like(A_hat)
        
        # set 1s at the top-k positions for each row
        batch_indices = torch.arange(A_hat.shape[0]).unsqueeze(1).expand(-1, k)
        A_hat[batch_indices, topk_indices] = 1

        # if A_hat is not symmetric, make it symmetric
        if not torch.allclose(A_hat, A_hat.T):
            # make the tensor symmetric: A_hat = (A_hat + A_hat.T) / 2
            A_hat = (A_hat + A_hat.T) / 2
            
            # convert to binary (in case averaging created 0.5 values) to ensure that the adjacency matrix is binary
            A_hat = torch.where(A_hat > 0, 1, 0)
    
    return A_hat