import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.utils import to_undirected, to_dense_adj


def sparse_mx_to_float_tensor(
        sparse_mx: sp.csr_matrix
    ) -> torch.Tensor:
    """
    Convert a scipy sparse matrix to a PyTorch float tensor.

    Parameters:
    ----------
    sparse_mx : scipy.sparse.csr_matrix
        A sparse matrix in Compressed Sparse Row format.

    Returns:
    --------
    torch.Tensor
        A dense PyTorch float tensor with dtype float32.
    """
    # Ensure the data is in float32 format directly
    sparse_mx = sparse_mx.astype(np.float32)

    # Convert directly to dense format as a PyTorch tensor
    dense_tensor = torch.tensor(sparse_mx.toarray(), dtype=torch.float32)
    return dense_tensor


def pandas_to_torch_one_hot(
        pandas_series: pd.Series,
        categories: list[str] | None = None
    ) -> torch.FloatTensor:
    """
    Convert a pandas Series of categorical labels to a PyTorch float tensor of one-hot encoded labels.

    Parameters:
    ----------
    pandas_series : pd.Series
        A pandas Series of categorical labels.
    categories : list[str]
        A list of categories for the categorical labels.

    Returns:
    --------
    torch.FloatTensor
        A PyTorch float tensor of one-hot encoded labels.
    """
    if categories is None:
        categories = sorted(pandas_series.unique())

    # Create a categorical series with all possible categories
    cat_series = pd.Categorical(pandas_series, categories=categories)

    # Convert to one-hot using all categories
    one_hot = pd.get_dummies(cat_series, prefix='', prefix_sep='')

    # Ensure all categories are present (even if they weren't in this batch)
    for cat in categories:
        if cat not in one_hot.columns:
            one_hot[cat] = 0

    # Sort columns to ensure consistent order
    one_hot = one_hot[sorted(one_hot.columns)]

    return torch.tensor(one_hot.values, dtype=torch.float32)


def edge_index_to_adjacency_tensor(
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
    """
    Convert an edge index to an adjacency tensor.

    Parameters:
    ----------
    edge_index: torch.Tensor
        The edge index to convert to an adjacency tensor.
        Dimensions: (2, num_edges)

    Returns:
    --------
    adjacency_matrix: torch.Tensor
        The adjacency tensor.
        Dimensions: (edge_index.max() + 1, edge_index.max() + 1)
    """
    # to_undirected ensures that the adjacency matrix is symmetric
    # to_dense_adj returns a tensor of shape (1, num_nodes, num_nodes)
    # here, num_nodes = edge_index.max() + 1
    adjacency_matrix = to_dense_adj(
                        to_undirected(edge_index).to(torch.long)
                        )[0]
    return adjacency_matrix