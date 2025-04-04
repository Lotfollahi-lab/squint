import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from typing import Optional
import scipy.sparse as sp

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


def compute_node_distributions(
        G: nx.Graph
    ) -> torch.Tensor:
    """
    Compute the local (node) clustering coefficient distribution given an adjacency matrix.

    Parameters
    ----------
    - adjacency_matrix: torch.Tensor
        The adjacency matrix.
        Dimensions: (num_nodes, num_nodes)

    Returns
    -------
    - node_degree_distribution: numpy.ndarray
        The node degree distribution.
        Dimensions: (num_nodes)
    - clustering_coefficient_distribution: numpy.ndarray
        The clustering coefficient distribution.
        Dimensions: (num_nodes)
    """
    # compute the node degree distribution
    node_degree_distribution = np.array(list(dict(G.degree()).values()))

    # compute the local clustering coefficient for each node
    clustering_coefficient_distribution = np.array(list(nx.clustering(G).values()))

    # compute the 4-orbit count distribution
    orbit_count_distribution = np.array(list(nx.four_cycles(G).values()))

    return node_degree_distribution, clustering_coefficient_distribution, orbit_count_distribution


def compute_spectral_density_distribution(
        G: nx.Graph,
        k: Optional[int] = 100,
        n_bins: Optional[int] = 50,
        density: Optional[bool] = True
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

    Returns
    -------
    - spectral_density: numpy.ndarray
        The spectral density distribution.
        Dimensions: (n_bins)
    """
    # compute the normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(G)

    # compute the eigenvalues of the Laplacian matrix
    # note: eigvalsh returns a 1D array of sorted eigenvalues with multiplicity
    # NOTE: maybe use shift-invert mode for better performance?
    eigenvalues = sp.eigsh(
                        A=L,
                        k=k,
                        which='SM',
                        tol=1e-4,
                        maxiter=1000,
                        return_eigenvectors=False,
                    )

    # quantize eigenvalues into a probability density function
    spectral_density, _ = np.histogram(
        a=eigenvalues,
        bins=n_bins,
        density=density,
    )

    return spectral_density


def compute_distribution_discrepancy(
        input_distribution: np.ndarray,
        target_distribution: np.ndarray,
        method: Optional[str] = 'total_variation'
    ) -> float:
    """
    Compute the discrepancy between two distributions.

    Parameters
    ----------
    - input_distribution: numpy.ndarray
        The input distribution.
        Dimensions: (num_nodes)
    - target_distribution: numpy.ndarray
        The target distribution.
        Dimensions: (num_nodes)
    - method: str
        The method to use to compute the discrepancy.
        Options: 'total_variation', 'kl_divergence'

    Returns
    -------
    - discrepancy: float
        The discrepancy between the two distributions.
    """
    if method == 'total_variation':
        return np.linalg.norm(
            x=input_distribution - target_distribution,
            ord=2
        )
    elif method == 'kl_divergence':
        return np.sum(
            a=input_distribution * np.log(input_distribution / target_distribution),
            axis=None,
        )
    else:
        raise ValueError(f"Invalid method: {method}")
