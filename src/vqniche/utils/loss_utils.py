import torch


def aggregate_1hop_neighbor_features(
        X: torch.Tensor,
        edge_index: torch.Tensor,
        return_mean: bool = False,
    ) -> torch.Tensor:
    """
    Compute the sum of the attribute vectors of the 1-hop neighbors of each node and return their mean if return_mean is True, otherwise return the sum.

    Parameters
    ----------
    - X: torch.Tensor
        The attribute vectors of the nodes.
        Dimensions: (num_nodes, num_features)
    - edge_index: torch.Tensor
        The edge index of the graph.
        Dimensions: (2, num_edges)
    - return_mean: bool
        Whether to return the mean or the sum of the neighbor features.

    Returns
    -------
    - X_nbr: torch.Tensor
        The mean or sum of the attribute vectors of the 1-hop neighbors of each node.
        Dimensions: (num_nodes, num_features)

    Notes
    -----
    - This assumes that all nodes have a self-loop.
    """
    # edge_index[0]: source nodes (j), edge_index[1]: target nodes (i)
    row, col = edge_index

    # Aggregate neighbor features: sum features of neighbors
    X_nbr = torch.zeros_like(X)
    X_nbr = X_nbr.index_add(0, col, X[row])

    # If return_mean is True, compute the mean of the neighbor features
    if return_mean:
        # Count neighbors
        deg = torch.zeros(X.shape[0], dtype=torch.float, device=X.device)
        deg = deg.index_add(0, col, torch.ones_like(col, dtype=torch.float))

        # Avoid division by zero
        deg = deg.clamp(min=1).unsqueeze(1)  # shape (n, 1)

        # Compute mean
        X_nbr = X_nbr / deg

    return X_nbr


def compute_neighbor_codebook_counts(
        indices: torch.Tensor,
        edge_index: torch.Tensor,
        codebook_size: int = 5000,
    ) -> torch.Tensor:
    """
    Compute count vector of codebook indices from 1-hop neighbors for each node.
    
    Parameters
    ----------
    - indices: torch.Tensor
        The codebook indices assigned to each node.
        Dimensions: (num_nodes,)
    - edge_index: torch.Tensor
        The edge index of the graph.
        Dimensions: (2, num_edges)
    - codebook_size: int
        Total number of codebook entries (default: 5000)
        
    Returns
    -------
    - neighbor_counts: torch.Tensor
        Count vector of neighbor codebook indices for each node.
        Dimensions: (num_nodes, codebook_size)
    """
    num_nodes = indices.shape[0]
    device = indices.device
    
    # edge_index[0]: source nodes (j), edge_index[1]: target nodes (i) 
    row, col = edge_index
    
    # Initialize count matrix
    neighbor_counts = torch.zeros(num_nodes, codebook_size, device=device, dtype=torch.float)
    
    # For each edge, increment the count of the source node's codebook index 
    # in the target node's neighbor count vector
    source_indices = indices[row]  # codebook indices of source nodes
    
    # Use scatter_add to accumulate counts - we need to expand for proper scatter
    neighbor_counts.index_add_(0, col, 
                              torch.nn.functional.one_hot(source_indices, num_classes=codebook_size).float())
    
    return neighbor_counts