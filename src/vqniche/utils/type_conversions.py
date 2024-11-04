import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer


def sparse_mx_to_float_tensor(sparse_mx: sp.csr_matrix) -> torch.FloatTensor:
    """
    Convert a scipy sparse matrix to a PyTorch float tensor.

    Args:
    ----
    sparse_mx : scipy.sparse.csr_matrix
        A sparse matrix in Compressed Sparse Row format.

    Returns:
    --------
    torch.FloatTensor
        A dense PyTorch float tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def pandas_to_torch_one_hot(pandas_series: pd.Series) -> torch.FloatTensor:
    """
    Convert a pandas Series of categorical labels to a PyTorch float tensor of one-hot encoded labels.

    Args:
    ----
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