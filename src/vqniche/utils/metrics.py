import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.stats import pearsonr
from typing import Optional, Tuple, Literal, List
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
        Dimensions: (batch, num_classes)
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


def compute_node_degree_distribution(
        G: nx.Graph,
    ) -> torch.Tensor:
    """
    Compute the local (node) degree distribution given a networkx graph.

    Parameters
    ----------
    - G: nx.Graph
        The networkx graph.

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
    node_degree_distribution = node_degree_distribution / node_degree_distribution.sum()

    return node_degree_distribution


def compute_node_clustering_coefficient_distribution(
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
        eigenvalues = np.linalg.eigvalsh(L.todense())[1:]
    else:
        # NOTE: maybe use shift-invert mode for better performance?
        eigenvalues = sp.linalg.eigsh(
                            A=L,
                            k=k,
                            which='SA',
                            tol=1e-4,
                            maxiter=200,
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


def compute_distribution_discrepancy(
        x: np.ndarray,
        y: np.ndarray,
        method: Literal['l1_gaussian_tv', 'l2_gaussian_tv'] = 'l1_gaussian_tv',
        sigma: Optional[float] = 1.0
    ) -> float:
    """
    Compute the discrepancy between two distributions.

    Parameters
    ----------
    - x: numpy.ndarray
        First distribution.
    - y: numpy.ndarray
        Second distribution.
    - method: Literal['l1_gaussian_tv', 'l2_gaussian_tv']
        The method to use to compute the discrepancy.
        Options: 'l1_gaussian_tv', 'l2_gaussian_tv'
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
    assert np.isclose(x.sum(), 1.0)
    assert np.isclose(y.sum(), 1.0)

    x_support = x.size
    y_support = y.size
    support = max(x_support, y_support)

    # pad the input distribution with zeros if it is smaller than the target distribution
    if x_support < support:
        x = np.hstack((
            x,
            np.zeros(support - x_support),
        ))

    # pad the target distribution with zeros if it is smaller than the input distribution
    if y_support < support:
        y = np.hstack((
            y,
            np.zeros(support - y_support),
        ))

    if method == 'l1_gaussian_tv':
        distance = np.abs(x - y).sum() / 2.0
        discrepancy = np.exp(-distance * distance / (2.0 * sigma ** 2))
    elif method == 'l2_gaussian_tv':
        distance = np.linalg.norm(x - y, ord=2)
        discrepancy = np.exp(-distance / (2.0 * sigma ** 2))

    return discrepancy


def compute_total_discrepancy(
        X: List[np.ndarray],
        Y: List[np.ndarray],
        method: Literal['l1_gaussian_tv', 'l2_gaussian_tv'] = 'l1_gaussian_tv',
        sigma: Optional[float] = 1.0
    ) -> np.ndarray:
    """
    Compute the total discrepancy between two collections of distributions.

    Parameters
    ----------
    - X: List[numpy.ndarray]
        First collection of distributions.
    - Y: List[numpy.ndarray]
        Second collection of distributions.
    - method: Literal['l1_gaussian_tv', 'l2_gaussian_tv']
        The method to use to compute the discrepancy.
        Options: 'l1_gaussian_tv', 'l2_gaussian_tv'
    - sigma: float
        The bandwidth of the Gaussian kernel.
    """
    total_discrepancy = 0.0
    for x in X:
        for y in Y:
            total_discrepancy += compute_distribution_discrepancy(
                                    x=x,
                                    y=y,
                                    method=method,
                                    sigma=sigma,
                                )
    total_discrepancy /= len(X) * len(Y)
    return total_discrepancy


def compute_mmd(
        D: List[np.ndarray],
        D_hat: List[np.ndarray],
        method: Literal['l1_gaussian_tv', 'l2_gaussian_tv'] = 'l1_gaussian_tv',
        sigma: Optional[float] = 1.0
    ) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two collections of distributions.

    Parameters
    ----------
    - D: List[numpy.ndarray]
        A collection of distributions, one per originalgraph.
    - D_hat: List[numpy.ndarray]
        A collection of distributions, one per reconstructed graph.
    - method: Literal['l1_gaussian_tv', 'l2_gaussian_tv']
        The method to use to compute the discrepancy.
        Options: 'l1_gaussian_tv', 'l2_gaussian_tv'
    - sigma: float
        The bandwidth of the Gaussian kernel.

    Returns
    -------
    - mmd: float
        The MMD between the two collections of distributions.
    """
    assert len(D) == len(D_hat)

    Kxx = compute_total_discrepancy(
        X=D,
        Y=D,
        method=method,
        sigma=sigma,
    )
    Kyy = compute_total_discrepancy(
        X=D_hat,
        Y=D_hat,
        method=method,
        sigma=sigma,
    )
    Kxy = compute_total_discrepancy(
        X=D,
        Y=D_hat,
        method=method,
        sigma=sigma,
    )
    print(f"Kxx: {Kxx}, Kyy: {Kyy}, Kxy: {Kxy}")
    mmd = Kxx + Kyy - 2 * Kxy
    return mmd


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
