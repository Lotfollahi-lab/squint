import torch
import torch.nn.functional as F


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


def compute_dispersion(
        input: torch.Tensor,
        num_out_features: int,
        theta: float = 1.0,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
    """
    Compute the dispersion of batch-ids for the negative binomial distribution.

    Parameters
    ----------
    input: torch.Tensor
        The input tensor.
        Dimensions: (batch_size, )
    num_out_features: int
        The number of output features of the dispersion.
    theta: float
        The scaling factor for weight initialization.
    device: torch.device
        The device to use for computation.

    Returns
    -------
    dispersion: torch.Tensor
        The computed dispersion.
        Dimensions: (batch_size, num_out_features)

    Notes
    -----
    - dispersion is computed as the exponential of the linear transformation of the one-hot encoding of the batch-ids.
    - `num_out_features` is the number of genes.
    - the 1D shape of `batch_ids` relies on the assumption that batch-ids are same for all the genes of a cell.
    """
    # Shape of one_hot currently: (batch_size, num_classes)
    one_hot = F.one_hot(
                    input,
                    num_classes=-1 # automatically infer the number of classes
                ).float().to(device)

    # If `input` is of shape (batch_size, *), then `one_hot` will be of shape (batch_size, *, num_classes)
    num_classes = one_hot.shape[-1]

    weight = torch.ones(
                    size=(num_out_features, num_classes),
                    dtype=torch.float32,
                    device=device
                ) * theta

    # Shape of F.linear: (batch_size, num_classes) x (num_out_features, num_classes).t() -> (batch_size, num_out_features)
    dispersions = F.linear(
        input = one_hot, # (batch_size, num_classes)
        weight=weight, # (num_out_features, num_classes)
    )

    # Shape of dispersions: (batch_size, num_out_features)
    return torch.exp(dispersions).to(device)