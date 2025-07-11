# Import metrics for adjacency
from .mmd import (
    mmd_score,
    degree_histogram,
    eigenvalues_pmf,
)

from .mlami import (
    compute_mlami,
)

from .gcs import (
    compute_gcs,
)

from .nasw import (
    compute_nasw,
)

from .clisis import (
    compute_clisis,
)

from .cas import (
    compute_cas,
)

from .benchmarking import (
    compute_benchmarking_metrics,
)

from .utils import (
    compute_knn_graph_connectivities_and_distances,
)

# Import metrics for attributes
from .pearson_correlation import (
    pearson_correlation,
)

from .cosine_similarity import (
    cosine_similarity,
)

# Import metrics for labels
from .accuracy_score import (
    accuracy_score,
)

# Define __all__ to specify what should be imported with "from vqniche.metrics import *"
__all__ = [
    # MMD functions
    "mmd_score",
    "degree_histogram",
    "eigenvalues_pmf",
    # Global spatial conservation metric (cell label supervised)
    "compute_mlami",
    "compute_gcs",
    "compute_nasw",
    "compute_clisis",
    "compute_cas",
    # Benchmarking functions
    "compute_benchmarking_metrics",
    # Attribute functions
    "pearson_correlation",
    "cosine_similarity",
    # Label functions
    "accuracy_score",
    # Utils functions
    "compute_knn_graph_connectivities_and_distances",
]
