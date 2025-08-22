import torch


def batch_pred_attr_and_target_attr(
            batch_x: torch.Tensor,
            batch_xhat: torch.Tensor,
            edge_index: torch.Tensor,
            batch_size: int,
            mask_idx: torch.Tensor,
            aggregate_1hop_neighbor_features: bool = False,
            only_masked: bool = False,
        ) -> torch.Tensor:
        """
        Prepare the predicted and target attributes for the loss computation.

        Parameters
        ----------
        - batch_x: torch.Tensor
            The original attributes of the batch.
            Dimensions: (num_nodes_in_batch, num_features)
        - batch_xhat: torch.Tensor
            The predicted attributes of the batch.
            Dimensions: (num_nodes_in_batch, num_features)
        - edge_index: torch.Tensor
            The edge index of the graph.
            Dimensions: (2, num_edges_in_batch)
        - batch_size: int
            The number of source nodes of the batch.
        - mask_idx: torch.Tensor
            The mask indices of the batch.
            Dimensions: (num_nodes_in_batch,)
        - aggregate_1hop_neighbor_features: bool
            Whether to aggregate the 1-hop neighbor features.
        - only_masked: bool
            Whether to only compute the loss over the masked nodes.

        Returns
        -------
        - pred_attr: torch.Tensor
            The predicted attributes of the batch.
            Dimensions: (batch_size, num_features) if only_masked is False, otherwise (mask_idx.sum(), num_features)
        - target_attr: torch.Tensor
            The target attributes of the batch.
            Dimensions: (batch_size, num_features) if only_masked is False, otherwise (mask_idx.sum(), num_features)

        Notes
        -----
        - This function is used to prepare the predicted and target attributes for the loss computation.
        - If aggregate_1hop_neighbor_features is True, the 1-hop neighbor features are aggregated.
        - If only_masked is True, the loss is computed over the masked nodes only.
        - If only_masked is False, the loss is computed over all nodes.
        """
        # 1) If aggregate_1hop_neighbor_features is True, use the 1-hop cell-wise micro-environment features
        if aggregate_1hop_neighbor_features:
            # 1.1) Imputed attributes from the decoder
            pred_attr = aggregate_1hop_neighbor_features(
                X=batch_xhat,
                edge_index=edge_index,
                return_mean=True,
            )
            # 1.2) Original attributes from the batch
            target_attr = aggregate_1hop_neighbor_features(
                X=batch_x,
                edge_index=edge_index,
                return_mean=True,
            )
        # 2) If aggregate_1hop_neighbor_features is False, use the cell-wise attributes
        else:
            pred_attr = batch_xhat
            target_attr = batch_x
        
        # 3) If only_masked is True, compute loss over masked nodes only
        if only_masked:
            # 3.1) If there are masked nodes, compute loss over masked nodes only
            if mask_idx.sum() > 0:
                pred_attr = pred_attr[mask_idx==1]
                target_attr = target_attr[mask_idx==1]
            # 3.2) If there are no masked nodes, compute loss over all nodes
            # this is to handle the case where the mask ratio is 0 in a given epochfor zeros and learnable_parameter mask strategies
            else:
                pred_attr = pred_attr[:batch_size]
                target_attr = target_attr[:batch_size]
        # 4) If only_masked is False, compute loss over all nodes
        elif not only_masked:
            pred_attr = pred_attr[:batch_size]
            target_attr = target_attr[:batch_size]

        return pred_attr, target_attr


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