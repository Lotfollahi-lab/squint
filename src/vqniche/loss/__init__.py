"""
Loss functions for VQNiche.

This module contains various loss functions used in the VQNiche model,
organized into separate files for better maintainability.
"""

from .cross_entropy import cross_entropy_loss
from .mse_attribute_reconstruction import mse_attribute_reconstruction_loss
from .nb_attribute_reconstruction import (
    nb_attribute_reconstruction_loss,
    nb_nbr_attribute_reconstruction_loss,
    nb_nbr_attribute_reconstruction_loss_dual,
)
from .mse_adjacency_reconstruction import mse_adjacency_reconstruction_loss
from .bce_adjacency_reconstruction import bce_adjacency_reconstruction_loss
from .bce_cosine_adjacency_reconstruction import bce_cosine_adjacency_reconstruction_loss
from .adversarial_batch import adversarial_batch_loss
from .spatial_prior_loss import ce_spatial_prior_loss
from .codebook_losses import (
    mse_commit_loss,
    mse_code_loss,
    mse_commit_loss_cell,
    mse_commit_loss_niche,
)
from .vqgraph_codebook_loss import (
    mse_joint_code_commit_loss,
    l2_codebook_orthogonal_regularization_loss
)
from .mask_token_regularization import mask_token_regularization
from .utils import compute_dispersion

__all__ = [
    "cross_entropy_loss",
    "mse_attribute_reconstruction_loss",
    "nb_attribute_reconstruction_loss",
    "nb_nbr_attribute_reconstruction_loss",
    "nb_nbr_attribute_reconstruction_loss_dual",
    "mse_adjacency_reconstruction_loss",
    "bce_adjacency_reconstruction_loss",
    "bce_cosine_adjacency_reconstruction_loss",
    "adversarial_batch_loss",
    "ce_spatial_prior_loss",
    "mse_commit_loss",
    "mse_code_loss",
    "mse_commit_loss_cell",
    "mse_commit_loss_niche",
    "mse_joint_code_commit_loss",
    "l2_codebook_orthogonal_regularization_loss",
    "mask_token_regularization",
    "compute_dispersion",
]
