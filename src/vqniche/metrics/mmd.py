from typing import Optional, Tuple, Literal, List

import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import scipy.sparse as sp

import torch


def compute_mmd_score(
        D: List[np.ndarray],
        D_hat: List[np.ndarray],
        method: Optional[Literal['basic', 'scipy']] = 'scipy',
    ) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two collections of distributions.

    Parameters
    ----------
    - D: List[numpy.ndarray]
        A collection of distributions.
    - D_hat: List[numpy.ndarray]
        A collection of distributions.

    Returns
    -------
    - mmd: float
        The MMD between the two collections of distributions.
    """
    assert len(D) == len(D_hat)

    if method == 'basic':
        compute_total_discrepancy = total_discrepancy
    elif method == 'scipy':
        compute_total_discrepancy = scipy_total_discrepancy
    
    total_discrepancy_by_bandwidth = []
    for bandwidth in [2, 1, 0.5, 0.01, 0.005]:
        K_XX = compute_total_discrepancy(
            X=D,
            Y=D,
            bandwidth=bandwidth,
        )
        K_YY = compute_total_discrepancy(
            X=D_hat,
            Y=D_hat,
            bandwidth=bandwidth,
        )
        K_XY = compute_total_discrepancy(
            X=D,
            Y=D_hat,
            bandwidth=bandwidth,
        )
        mmd_bandwidth = K_XX + K_YY - 2 * K_XY
        total_discrepancy_by_bandwidth.append(mmd_bandwidth)

    mmd = np.mean(total_discrepancy_by_bandwidth)

    return mmd


def scipy_total_discrepancy(
        X: List[np.ndarray],
        Y: List[np.ndarray],
        kernel: Optional[Literal['l1_gaussian_tv']] = 'l1_gaussian_tv',
        bandwidth: float = 1.0,
        dtype=np.float32,
    ) -> float:
    assert kernel == 'l1_gaussian_tv'
    maxw = max(max(x.size for x in X), max(y.size for y in Y))
    Xp = np.zeros((len(X), maxw), dtype=dtype)
    Yp = np.zeros((len(Y), maxw), dtype=dtype)
    for i, a in enumerate(X): Xp[i, :a.size] = a
    for j, b in enumerate(Y): Yp[j, :b.size] = b

    if kernel == 'l1_gaussian_tv':
        D = cdist(Xp, Yp, metric='cityblock')          # L1 distances (n, m)
        K = np.exp(-(D * D) / (2.0 * (bandwidth ** 2)))

    return float(K.mean())


def total_discrepancy(
        X: List[np.ndarray],
        Y: List[np.ndarray],
    ) -> np.ndarray:
    """
    Compute the total discrepancy between two collections of distributions.

    Parameters
    ----------
    - X: List[numpy.ndarray]
        First collection of distributions.
    - Y: List[numpy.ndarray]
        Second collection of distributions.
    """
    total_discrepancy = 0.0
    for x in X:
        for y in Y:
            total_discrepancy += distribution_discrepancy(
                                    x=x,
                                    y=y,
                                )
    total_discrepancy /= len(X) * len(Y)
    return total_discrepancy


def distribution_discrepancy(
        x: np.ndarray,
        y: np.ndarray,
        kernel: Optional[Literal['l1_gaussian_tv']] = 'l1_gaussian_tv',
        bandwidth: Optional[float] = 1.0,
    ) -> float:
    """
    Compute the discrepancy between two distributions.

    Parameters
    ----------
    - x: numpy.ndarray
        First distribution.
    - y: numpy.ndarray
        Second distribution.
    - kernel: Literal['l1_gaussian_tv', 'l2_gaussian_tv']
        The kernel to use to compute the discrepancy.
        Options: 'l1_gaussian_tv', 'l2_gaussian_tv'
    - bandwidth: float
        The bandwidth of the Gaussian kernel.

    Returns
    -------
    - discrepancy: float
        The discrepancy between the two distributions.

    References
    ----------
    - https://github.com/KarolisMart/SPECTRE/blob/main/util/dist_helper.py
    """
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

    if kernel == 'l1_gaussian_tv':
        distance = np.abs(x - y).sum() / 2.0
        discrepancy = np.exp(-distance * distance / (2.0 * bandwidth ** 2))

    return discrepancy


def degree_histogram(
        G: nx.Graph,
    ) -> torch.Tensor:
    """
    Compute the normalized degree histogram of a graph.

    Parameters
    ----------
    - G: nx.Graph
        graph.

    Returns
    -------
    - degree_histogram: numpy.ndarray
        The normalized degree histogram.
        Dimensions: (num_unique_degrees + 1,)

    References
    ----------
    - Adapted from https://github.com/KarolisMart/SPECTRE/blob/main/util/eval_helper.py
    """
    # compute the degree histogram
    degree_histogram = np.array(nx.degree_histogram(G))

    # normalize the degree histogram
    degree_histogram = degree_histogram / degree_histogram.sum()

    return degree_histogram


def eigenvalues_pmf(
        G: nx.Graph,
        k: int = 250,
        n_bins: int = 50,
    ) -> np.ndarray:
    """
    Quantize eigenvalues of the normalized Laplacian matrix into a probability mass function via binning.

    Parameters
    ----------
    - G: nx.Graph
        The graph.
    - k: int
        The number of eigenvalues to compute.
    - n_bins: int
        The number of bins to use for the histogram.

    Returns
    -------
    - eigenvalues_pmf: numpy.ndarray
        Probability mass function of Laplacian eigenvalues.
        Dimensions: (n_bins,)

    References
    ----------
    - Adapted from https://github.com/KarolisMart/SPECTRE/blob/main/util/eval_helper.py
    """
    # compute the normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(G)

    # compute the k (algebraically) smallest eigenvalues of the Laplacian matrix
    eigenvalues = sp.linalg.eigsh(
                        A=L,
                        k=k,
                        which='SA',
                        tol=1e-4,
                        maxiter=200,
                        return_eigenvectors=False,
                )

    # quantize eigenvalues into a probability mass function via binning
    return pmf_via_binning(
        input=eigenvalues,
        n_bins=n_bins,
        range=(-1e-6, 2),
        normalize=True,
    )


def pmf_via_binning(
        input: np.ndarray,
        n_bins: Optional[int] = 50,
        range: Optional[Tuple[float, float]] = None,
        normalize: Optional[bool] = True
    ) -> np.ndarray:
    """
    Compute the probability mass function of a given input array via binning.

    Parameters
    ----------
    - input: numpy.ndarray
        The input array.
    - n_bins: int
        The number of bins to use for the histogram.
    - range: tuple
        The range of the histogram.
    - normalize: bool
        Whether to normalize the histogram.

    Returns
    -------
    - pmf: numpy.ndarray
        The probability mass function of the input array.
        Dimensions: (n_bins,)
    """
    pmf, _ = np.histogram(
        a=input,
        bins=n_bins,
        range=range,
        density=False,
    )
    if normalize:
        pmf = pmf / pmf.sum()

    return pmf
