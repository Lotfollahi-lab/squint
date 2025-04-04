import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Optional, Tuple, Literal
from sklearn.metrics import accuracy_score as sklearn_accuracy_score

import torch
import torch.nn.functional as F


def accuracy_score(
        unnormalized_logits: torch.Tensor,
        one_hot_labels: torch.Tensor
    ) -> float:
    """
    Compute the accuracy score for a given set of unnormalized logits and labels.

    Parameters
    ----------
    - logits: torch.Tensor
        The unnormalized logits.
        Dimensions: (batch_size, num_classes)
    - labels: torch.Tensor
        The true labels.
        Dimensions: (batch_size, num_classes)

    Returns
    -------
    - accuracy: float
        The accuracy score.
    """
    # compute the predicted class probabilities (normalized logits)
    normalized_logits = unnormalized_logits.softmax(dim=-1)

    # convert to predicted class indices in a numpy array
    predicted_labels = np.argmax(
                            normalized_logits.detach().cpu().numpy(),
                            axis=1,
                        )

    # convert to true class indices in a numpy array
    true_labels = np.argmax(
                            one_hot_labels.detach().cpu().numpy(),
                            axis=1,
                        )

    # compute the accuracy score
    accuracy = sklearn_accuracy_score(
                    y_true=true_labels,
                    y_pred=predicted_labels,
                )

    return accuracy


@staticmethod
def get_similarity_stats(
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


def build_reconstructed_adjacency_matrix(
        pred_adj: torch.Tensor,
    ) -> torch.Tensor:
    """
    Build the reconstructed adjacency matrix from the predicted adjacency matrix and the edge index.

    Parameters
    ----------
    - pred_adj: torch.Tensor
        The quantized matrix coming from the decoder module.
        Dimensions: (num_nodes, decoder_embedding_dim)

    Returns
    -------
    - reconstructed_adjacency_matrix: torch.Tensor
        The reconstructed adjacency matrix.
        Dimensions: (num_nodes, num_nodes)
    """
    reconstr_adj = torch.matmul(pred_adj.detach(), pred_adj.detach().t())
    reconstr_adj = (reconstr_adj - reconstr_adj.min()) / (reconstr_adj.max() - reconstr_adj.min() + 1e-8)

    reconstr_adj[reconstr_adj < 0.5] = 0
    reconstr_adj[reconstr_adj >= 0.5] = 1

    return reconstr_adj


def compute_distribution(
        input: np.ndarray,
        n_bins: Optional[int] = 50,
        range: Optional[Tuple[float, float]] = None,
        density: Optional[bool] = False,
        normalize: Optional[bool] = True
    ) -> np.ndarray:
    """
    Compute the distribution of a given input array.

    Parameters
    ----------
    - input: numpy.ndarray
        The input array.
    - n_bins: int
        The number of bins to use for the histogram.
    - density: bool
        Whether to return the density of the histogram.
    - normalize: bool
        Whether to normalize the histogram.

    Returns
    -------
    - distribution: numpy.ndarray
        The distribution of the input array.
        Dimensions: (n_bins)
    """
    distribution, _ = np.histogram(
        a=input,
        bins=n_bins,
        range=range,
        density=density,
    )
    if normalize:
        distribution = distribution / distribution.sum()

    return distribution


def node_degree_distribution(
        G: nx.Graph,
        n_bins: Optional[int] = 100,
        density: Optional[bool] = False,
        normalize: Optional[bool] = True
    ) -> torch.Tensor:
    """
    Compute the local (node) degree distribution given a networkx graph.

    Parameters
    ----------
    - G: nx.Graph
        The networkx graph.
    - n_bins: int
        The number of bins to use for the histogram.
    - density: bool
        Whether to return the density of the histogram.
    - normalize: bool
        Whether to normalize the histogram.

    Returns
    -------
    - node_degree_distribution: numpy.ndarray
        The node degree distribution.
        Dimensions: (num_nodes)

    References
    ----------
    - Adapted from https://github.com/KarolisMart/SPECTRE/blob/main/util/eval_helper.py
    """
    # compute the node degree distribution
    node_degree_distribution = np.array(nx.degree_histogram(G))

    return compute_distribution(
        input=node_degree_distribution,
        n_bins=n_bins,
        range=(0.0, 1.0),
        density=density,
        normalize=normalize,
    )


def node_clustering_coefficient_distribution(
        G: nx.Graph,
        n_bins: Optional[int] = 100,
        density: Optional[bool] = False,
        normalize: Optional[bool] = True
    ) -> np.ndarray:
    """
    Compute the local clustering coefficient distribution for a given networkx graph.

    Parameters
    ----------
    - G: nx.Graph
        The networkx graph.
    - n_bins: int
        The number of bins to use for the histogram.
    - density: bool
        Whether to return the density of the histogram.
    - normalize: bool
        Whether to normalize the histogram.

    Returns
    -------
    - node_cc_distribution: numpy.ndarray
        The local clustering coefficient distribution.
        Dimensions: (num_nodes)

    References
    ----------
    - Adapted from https://github.com/KarolisMart/SPECTRE/blob/main/util/eval_helper.py
    """
    # compute the local clustering coefficient for each node
    node_cc = np.array(list(nx.clustering(G).values()))

    return compute_distribution(
        input=node_cc,
        n_bins=n_bins,
        range=(0.0, 1.0),
        density=density,
        normalize=normalize,
    )


def compute_spectral_distribution(
        G: nx.Graph,
        k: Optional[int] = None,
        n_bins: Optional[int] = 50,
        density: Optional[bool] = False,
        normalize: Optional[bool] = True
    ) -> np.ndarray:
    """
    Compute the spectral density distribution for a given graph.

    Parameters
    ----------
    - G: nx.Graph
        The graph.
    - k: int
        The number of eigenvalues to compute.
    - n_bins: int
        The number of bins to use for the histogram.
    - density: bool
        Whether to return the density of the histogram.
    - normalize: bool
        Whether to normalize the histogram.

    Returns
    -------
    - spectral_distribution: numpy.ndarray
        The spectral distribution.
        Dimensions: (n_bins)

    References
    ----------
    - Adapted from https://github.com/KarolisMart/SPECTRE/blob/main/util/eval_helper.py
    """
    # compute the normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(G)

    # compute the eigenvalues of the Laplacian matrix
    # note: eigvalsh returns a 1D array of sorted eigenvalues with multiplicity
    if k is None:
        eigenvalues = np.eigvalsh(L.todense())[1:]
    else:
        # NOTE: maybe use shift-invert mode for better performance?
        eigenvalues = sp.eigsh(
                            A=L,
                            k=k,
                            which='SA',
                            tol=1e-4,
                            maxiter=1000,
                            return_eigenvectors=False,
                    )

    # quantize eigenvalues into a probability mass function
    return compute_distribution(
        input=eigenvalues,
        n_bins=n_bins,
        range=(-1e-6, 2),
        density=density,
        normalize=normalize,
    )


def compute_mmd(
        input_distribution: np.ndarray,
        target_distribution: np.ndarray,
        method: Literal['gaussian_tv'] = 'gaussian_tv',
        sigma: Optional[float] = 1.0
    ) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions.

    Parameters
    ----------
    - input_distribution: numpy.ndarray
        The input distribution.
        Dimensions: (num_nodes)
    - target_distribution: numpy.ndarray
        The target distribution.
        Dimensions: (num_nodes)
    - method: Literal['gaussian_tv']
        The method to use to compute the discrepancy.
        Options: 'gaussian_tv'
    - sigma: float
        The bandwidth of the Gaussian kernel.

    Returns
    -------
    - discrepancy: float
        The discrepancy between the two distributions.

    References
    ----------
    - https://github.com/KarolisMart/SPECTRE/blob/main/util/dist_helper.py
    """
    assert input_distribution.sum() == 1.0
    assert target_distribution.sum() == 1.0

    input_support_size = input_distribution.size
    target_support_size = target_distribution.size
    support_size = max(input_support_size, target_support_size)

    # pad the input distribution with zeros if it is smaller than the target distribution
    if input_support_size < support_size:
        input_distribution = np.pad(
            array=input_distribution,
            pad_width=((0, 0), (0, support_size - input_support_size)),
            mode='constant',
            constant_values=0.0,
        )

    # pad the target distribution with zeros if it is smaller than the input distribution
    if target_support_size < support_size:
        target_distribution = np.pad(
            array=target_distribution,
            pad_width=((0, 0), (0, support_size - target_support_size)),
            mode='constant',
            constant_values=0.0,
        )

    if method == 'gaussian_tv':
        distance = np.abs(input_distribution - target_distribution).sum() / 2.0
        discrepancy = np.exp(-distance / (2.0 * sigma ** 2))

    return discrepancy


def compute_pearson_correlation(
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
    if compare_genes:
        pearson_correlation = np.corrcoef(X, X_hat)[0, 1]
    else:
        pearson_correlation = np.corrcoef(X, X_hat)[1, 0]

    if mean:
        return pearson_correlation.mean()
    else:
        return pearson_correlation
