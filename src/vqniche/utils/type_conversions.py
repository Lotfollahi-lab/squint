import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.utils import to_scipy_sparse_matrix


def sparse_mx_to_float_tensor(sparse_mx: sp.csr_matrix) -> torch.Tensor:
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


def pandas_to_torch_one_hot(pandas_series: pd.Series) -> torch.FloatTensor:
    """
    Convert a pandas Series of categorical labels to a PyTorch float tensor of one-hot encoded labels.

    Parameters:
    ----------
    pandas_series : pd.Series
        A pandas Series of categorical labels.

    Returns:
    --------
    torch.FloatTensor
        A PyTorch float tensor of one-hot encoded labels.
    """
    numerical_labels = pandas_series.astype('category').values.codes
    one_hot_labels = LabelBinarizer().fit_transform(numerical_labels)
    return torch.tensor(one_hot_labels, dtype=torch.float32)


def edge_index_to_adjacency_tensor(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device
    ) -> torch.Tensor:
    """
    Convert an edge index to an adjacency tensor.

    Parameters:
    ----------
    edge_index: torch.Tensor
        The edge index to convert to an adjacency tensor.
    num_nodes: int
        The number of nodes in the graph.
    device: torch.device
        The device to use for the adjacency tensor.

    Returns:
    --------
    torch.Tensor
        The adjacency tensor.
    """
    adjacency_matrix = to_scipy_sparse_matrix(
                        edge_index,
                        num_nodes=num_nodes
                        ).toarray()
    adjacency_tensor = sparse_mx_to_float_tensor(
                        sp.csr_matrix(adjacency_matrix)
                        ).to(device)
    return adjacency_tensor