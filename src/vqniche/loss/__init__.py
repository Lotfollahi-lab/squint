"""
Loss functions for VQNiche.

This module contains various loss functions used in the VQNiche model,
organized into separate files for better maintainability.
"""

from .cross_entropy import cross_entropy
from .mse_attribute_reconstruction import mse_attribute_reconstruction
from .nb_attribute_reconstruction import nb_attribute_reconstruction
from .mse_adjacency_reconstruction import mse_adjacency_reconstruction
from .codebook_losses import (
    mse_commit_loss,
    mse_code_loss
)
from .vqgraph_codebook_loss import (
    mse_joint_code_commit_loss,
    l2_codebook_orthogonal_regularization_loss
)
from .utils import compute_dispersion

__all__ = [
    "cross_entropy",
    "mse_attribute_reconstruction", 
    "nb_attribute_reconstruction",
    "mse_adjacency_reconstruction",
    "mse_commit_loss",
    "mse_code_loss",
    "mse_joint_code_commit_loss",
    "l2_codebook_orthogonal_regularization_loss",
    "compute_dispersion"
] 