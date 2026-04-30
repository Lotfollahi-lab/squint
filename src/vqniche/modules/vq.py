from typing import Dict, Any

import torch
import inspect
from vector_quantize_pytorch import (
    VectorQuantize,
    ResidualVQ,
    GroupedResidualVQ,
    SimVQ,
    FSQ,
    ResidualFSQ,
    LFQ,
    ResidualLFQ,
    LatentQuantize,
    RandomProjectionQuantizer
)

# Local hierarchical / tree-structured VQ variants
from .hierarchical_vq import ResidualVQ_Squint, ConditionalVQ


def get_vq_class(
        vq_name: str,
    ) -> torch.nn.Module:
    """
    Get the VQ class from its name.

    Parameters
    ----------
    - vq_name: str
        The name of the VQ class to instantiate.

    Returns
    -------
    - torch.nn.Module
        The VQ class.

    Notes
    -----
    - The VQ classes are the ones available in the `vector_quantize_pytorch` package.
    - Reference: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/README.md
    """
    vq_classes = {
        'VectorQuantize': VectorQuantize,
        'ResidualVQ': ResidualVQ,
        'GroupedResidualVQ': GroupedResidualVQ,
        'SimVQ': SimVQ,
        'FSQ': FSQ,
        'ResidualFSQ': ResidualFSQ,
        'LFQ': LFQ,
        'ResidualLFQ': ResidualLFQ,
        'LatentQuantize': LatentQuantize,
        'RandomProjectionQuantizer': RandomProjectionQuantizer,
        # Local hierarchical variants (in-tree, built atop VectorQuantize):
        'ResidualVQ_Squint': ResidualVQ_Squint,   # RQ-VAE with per-level sizing
        'ConditionalVQ':    ConditionalVQ,         # tree (level-1 routes to level-2)
    }

    if vq_name not in vq_classes:
        raise ValueError(f"Unknown VQ class name: {vq_name}. Must be one of {list(vq_classes.keys())}")

    return vq_classes[vq_name]


def get_valid_params(
        cls: torch.nn.Module,
        params_dict: dict,
    ) -> Dict[str, Any]:
    """
    Filter out only the parameters in the class's __init__ signature,
    check for required parameters, and handle special cases.

    Returns
    -------
    - Dict[str, Any]
        The filtered dictionary of parameters that are valid for the VQ class.
    Raises
    ------
    - ValueError if required parameters are missing.
    """
    required_params = {
        'VectorQuantize': ['dim', 'codebook_size'],
        'ResidualVQ': ['dim', 'num_quantizers', 'codebook_size'],
        'GroupedResidualVQ': ['dim', 'num_quantizers', 'groups', 'codebook_size'],
        'SimVQ': ['dim', 'codebook_size'],
        'FSQ': ['levels'],
        'ResidualFSQ': ['dim', 'levels', 'num_quantizers'],
        'LFQ': [],  # Either dim or codebook_size must be specified
        'ResidualLFQ': ['dim', 'codebook_size', 'num_quantizers'],
        'LatentQuantize': ['levels', 'dim'],
        'RandomProjectionQuantizer': ['dim', 'num_codebooks', 'codebook_dim', 'codebook_size'],
        # Local hierarchical variants — required params match their __init__:
        'ResidualVQ_Squint': ['dim', 'num_quantizers', 'codebook_size'],
        'ConditionalVQ':    ['dim', 'codebook_size_l1', 'codebook_size_l2'],
    }
    class_name = cls.__name__
    init_signature = inspect.signature(cls.__init__)
    valid_param_names = set(init_signature.parameters.keys())

    # Filter only valid params
    valid_params = {k: v for k, v in params_dict.items() if k in valid_param_names}

    if class_name == 'VectorQuantize':
        if 'in_place_codebook_optimizer' in valid_params:
            if params_dict['in_place_codebook_optimizer'] == 'adam':
                valid_params['in_place_codebook_optimizer'] = torch.optim.Adam
            elif params_dict['in_place_codebook_optimizer'] == 'sgd':
                valid_params['in_place_codebook_optimizer'] = torch.optim.SGD
            elif params_dict['in_place_codebook_optimizer'] is None:
                valid_params['in_place_codebook_optimizer'] = None
            else:
                raise ValueError(f"Invalid in_place_codebook_optimizer: {params_dict['in_place_codebook_optimizer']}. Must be one of 'adam' or 'sgd'.")
        else:
            valid_params['in_place_codebook_optimizer'] = None

    # Special case for LFQ
    if class_name == 'LFQ':
        if not ('dim' in valid_params or 'codebook_size' in valid_params):
            raise ValueError("LFQ requires either 'dim' or 'codebook_size' parameter")
    else:
        missing = [p for p in required_params[class_name] if p not in valid_params]
        if missing:
            raise ValueError(f"Missing required parameters for {class_name}: {missing}")

    return valid_params