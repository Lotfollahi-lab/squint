"""
VQNiche utilities package.

This package contains various utility functions and classes for the VQNiche project, including type conversions, adjacency reconstruction, loss utilities, masking for imputation, and configuration parsing.
"""

# Type conversion utilities
from .type_conversions import (
    sparse_mx_to_float_tensor,
    pandas_to_torch_one_hot,
    one_dim_to_one_hot,
    edge_index_to_adjacency_tensor,
    adjacency_tensor_to_networkx,
    torch_one_hot_to_label_name,
    data_batch_to_adata_list,
    inference_data_dict_to_adata,
)

# Adjacency reconstruction utilities
from .adjacency_reconstruction import (
    reconstruct_adjacency_matrix,
)

# Loss utilities
from .loss_utils import (
    batch_pred_attr_and_target_attr,
    aggregate_1hop_neighbor_features,
    compute_neighbor_codebook_counts,
)

# Masking utilities
from .mask import (
    set_mask_ratio,
    set_mask_indices,
    print_masked_input_diversity_stats,
)

# Configuration parsing utilities
from .parse_train_configs import (
    parse_train_arguments,
    collect_train_configs,
    prepare_sweep_config,
    update_config,
)

from .parse_test_configs import (
    parse_test_arguments,
    collect_test_configs,
    find_best_checkpoint,
)

from .parse_datasetblob_configs import (
    parse_datasetblob_arguments,
    collect_datasetblob_configs,
)

# Export all public functions and classes
__all__ = [
    # Type conversions
    "sparse_mx_to_float_tensor",
    "pandas_to_torch_one_hot", 
    "one_dim_to_one_hot",
    "edge_index_to_adjacency_tensor",
    "adjacency_tensor_to_networkx",
    "torch_one_hot_to_label_name",
    "data_batch_to_adata_list",
    "inference_data_dict_to_adata",
    
    # Adjacency reconstruction
    "reconstruct_adjacency_matrix",
    
    # Loss utilities
    "batch_pred_attr_and_target_attr",
    "aggregate_1hop_neighbor_features",
    "compute_neighbor_codebook_counts",
    
    # Masking utilities
    "set_mask_ratio",
    "set_mask_indices",
    "print_masked_input_diversity_stats",
    
    # Configuration parsing
    "parse_train_arguments",
    "collect_train_configs",
    "prepare_sweep_config", 
    "update_config",
    "parse_test_arguments",
    "collect_test_configs",
    "find_best_checkpoint",
    "parse_datasetblob_arguments",
    "collect_datasetblob_configs",
]
