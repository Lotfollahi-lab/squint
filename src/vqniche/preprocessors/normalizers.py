import torch
import scipy.sparse as sp
from typing import Union
import scanpy as sc


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
    x_hat = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return x_hat