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