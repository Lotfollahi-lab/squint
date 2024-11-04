import torch
import scipy.sparse as sp
from typing import Union


def normalize_by_read_depth(x: Union[sp.csr_matrix , torch.Tensor],
                            target_size: int=10_000,
                            ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by read depth.

    Parameters
    ----------
    x : sp.csr_matrix
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    target_size : int
        The target read depth per observation (i.e. the sum of features across
        an observation).

    Returns
    -------
    sp.csr_matrix | torch.Tensor :
        A sparse matrix or Tensor containing the normalized features.

    Reference:
    ----------
    NicheJEPA (Author: Sebastian Birk)
    """
    if isinstance(x, sp.csr_matrix):
        x_hat = x / x.sum(axis=1).reshape(-1, 1) * target_size
    elif isinstance(x, torch.Tensor):
        # Convert sparse tensor to COO format if it's not already
        x = x.coalesce()

        # Extract indices and values
        indices = x.indices()  # Shape: (2, nnz), where nnz is the number of non-zero elements
        values = x.values()    # Shape: (nnz,)

        # Compute the sum of each row
        row_sums = torch.sparse.sum(x, dim=1).to_dense()  # Dense vector of row sums, shape: (num_rows,)

        # Gather row sums corresponding to each non-zero element
        row_indices = indices[0]  # Row indices of each non-zero element
        row_sum_values = row_sums[row_indices]  # Shape: (nnz,)

        # Normalize values by the corresponding row sum
        normalized_values = values / row_sum_values

        # Scale the normalized values by target_size
        scaled_values = normalized_values * target_size

        # Create a new sparse tensor with the same indices but scaled values
        x_hat = torch.zeros(x.size(), dtype=scaled_values.dtype)
        x_hat[indices[0], indices[1]] = scaled_values

    return x_hat