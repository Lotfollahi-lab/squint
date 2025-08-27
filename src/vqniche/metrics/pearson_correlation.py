from typing import Optional

import numpy as np
from scipy.stats import pearsonr
from anndata import AnnData

from ..utils.loss_utils import aggregate_1hop_neighbor_features


def compute_pearson_correlation(
        adata: AnnData,
        X_key: str = 'X',
        X_hat_key: str = 'X_hat',
        nbr_key: Optional[str] = None,
        compare_genes: Optional[bool] = False,
    ) -> float:
    """
    Compute the Pearson correlation between original and reconstructed gene expressions.
    If nbr_key is provided, the Pearson correlation is computed between the 1-hop neighbor-wise gene expressions.

    Parameters
    ----------
    - adata: AnnData
        The AnnData object containing the original and reconstructed gene expressions.
    - X_key: str
        The key in adata.obsm where the original gene expressions are stored.
    - X_hat_key: str
        The key in adata.obsm where the reconstructed gene expressions are stored.
    - nbr_key: Optional[str]
        The key in adata.obsm where the edge index is stored.
        If provided, the Pearson correlation is computed between the 1-hop neighbor-wise gene expressions.
    - compare_genes: Optional[bool]
        Whether to compare the genes of the two matrices.
        If True, the Pearson correlation is computed between the genes of the two matrices.
        If False, the Pearson correlation is computed between the cells of the two matrices.

    Returns
    -------
    - pearson_correlation: float
        The Pearson correlation between the original and reconstructed gene expressions.
        If nbr_key is provided, the Pearson correlation is computed between the 1-hop neighbor-wise gene expressions.
    
    Notes
    -----
    - This function only supports using the same edge index for the original and reconstructed gene expressions of the 1 hop neighborhood.
    - 
    - Currently, this function only supports using the same edge index for the original and reconstructed gene expressions of the 1 hop neighborhood. This may need to be updated to support the original and reconstructed edge indices.
    """
    if nbr_key is not None:
        X_nbr = aggregate_1hop_neighbor_features(
                    X=adata.uns[X_key],
                    edge_index=adata.uns[nbr_key],
                    return_mean=True,
                )
        X_hat_nbr = aggregate_1hop_neighbor_features(
                    X=adata.uns[X_hat_key],
                    edge_index=adata.uns[nbr_key],
                    return_mean=True,
                )
        return pearson_correlation(
            X=X_nbr,
            X_hat=X_hat_nbr,
            compare_genes=compare_genes,
            mean=True,
        )
    else:
        return pearson_correlation(
            X=adata.uns[X_key],
            X_hat=adata.uns[X_hat_key],
            compare_genes=compare_genes,
            mean=True,
        )


def pearson_correlation(
        X: np.ndarray,
        X_hat: np.ndarray,
        compare_genes: Optional[bool] = False,
        mean: Optional[bool] = True
    ) -> float:
    """
    Compute the Pearson correlation between original and reconstructed gene expressions.

    Parameters
    ----------
    - X: numpy.ndarray
        The original cell-gene matrix.
    - X_hat: numpy.ndarray
        The reconstructed cell-gene matrix.
    - compare_genes: bool
        Whether to compare the genes of the two matrices.
        If True, the Pearson correlation is computed between the genes of the two matrices.
        If False, the Pearson correlation is computed between the cells of the two matrices.
    - mean: bool
        Whether to return the mean Pearson correlation.
        If True, the mean Pearson correlation is returned.
        If False, the Pearson correlation is returned for each gene/cell.

    Returns
    -------
    - pearson_correlation: float | numpy.ndarray
        The mean Pearson correlation
        or the Pearson correlation for each gene/cell.
    """
    assert X.shape == X_hat.shape

    if compare_genes:
        correlations = [pearsonr(X[:, j], X_hat[:, j])[0] for j in range(X.shape[1])]
    else:
        correlations = [pearsonr(X[i, :], X_hat[i, :])[0] for i in range(X.shape[0])]
    correlations = np.array(correlations)

    if mean:
        return correlations.mean()
    else:
        return correlations