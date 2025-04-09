import scanpy as sc
import anndata as ad
import numpy as np
from typing import Union

import torch
import scipy.sparse as sp


def normalize_by_read_depth(
        x: Union[sp.csr_matrix , torch.Tensor],
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
    NEMO (Author: Sebastian Birk)
    """
    x_hat = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return x_hat


def normalize_total_log1p(
        x: Union[sp.csr_matrix , torch.Tensor],
        target_size: int=10_000,
        apply_CPM: bool=True,
    ) -> sp.csr_matrix:
    """
    Normalize counts per cell by total counts over all genes and log1p transform.

    Parameters
    ----------
    x : sp.csr_matrix
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    target_size : int
        The target read depth per observation (i.e. the sum of features across
        an observation).
    apply_CPM : bool
        If True, apply CPM normalization.

    Returns
    -------
    sp.csr_matrix :
        A sparse matrix containing the normalized features.
    """
    if apply_CPM:
        adata = ad.AnnData(x.todense().astype(float))
        x = sc.pp.normalize_total(
                adata,
                target_sum=target_size,
                inplace=False
            )['X']
    x = sp.csr_matrix(np.log1p(x))
    return x