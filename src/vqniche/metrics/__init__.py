# Import functions from mmd.py that are used in the models
from .mmd import (
    mmd_score,
    degree_histogram,
    eigenvalues_pmf,
)

# Import functions from attributes.py that are used in the models
from .attributes import (
    pearson_correlation,
    cosine_similarity,
)

# Import functions from labels.py that are used in the models
from .labels import (
    accuracy_score,
)

# Define __all__ to specify what should be imported with "from vqniche.metrics import *"
__all__ = [
    # MMD functions
    "mmd_score",
    "degree_histogram",
    "eigenvalues_pmf",
    # Attribute functions
    "pearson_correlation",
    "cosine_similarity",
    # Label functions
    "accuracy_score",
]
