from typing import Optional

import numpy as np
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F


def pearson_correlation(
        X: np.ndarray,
        X_hat: np.ndarray,
        compare_genes: Optional[bool] = False,
        mean: Optional[bool] = True
    ) -> float:
    """
    Compute the Pearson correlation between original and reconstructed cell-gene matrices.

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


def cosine_similarity(
        embeddings: torch.Tensor,
        prefix: str
    ) -> dict:
    """
    Compute pairwise similarity statistics for all embeddings.

    Returns
    -------
    - similarity_stats: dict
        Dictionary containing mean and std of pairwise cosine similarities for different embeddings
    """
    N = embeddings.size(0)
    # Normalize embeddings for cosine similarity
    normalized = F.normalize(embeddings, p=2, dim=1)

    # Initialize storage for upper triangle similarities
    n_pairs = (N * (N - 1)) // 2
    similarities = torch.empty(n_pairs, device=embeddings.device)

    # Compute only upper triangle elements
    idx = 0
    for i in range(N-1):
        # Compute similarity between embedding i and all j > i
        sims = torch.matmul(normalized[i:i+1], normalized[i+1:].t())
        similarities[idx:idx+N-i-1] = sims[0]
        idx += N-i-1

    return {
        f'{prefix}_mean': similarities.mean().item(),
    }