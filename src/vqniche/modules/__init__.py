from .mlp import MLP
from .conditional_mlp import ConditionalMLP
from .gnn import init_gnn_module, create_dynamic_gnn_module_class
from .cosine_codebook import CosineSimCodebook
from .vq import get_vq_class, get_valid_params
from .hierarchical_vq import ResidualVQ_Squint, ConditionalVQ
from .dispersion_head import DispersionHead
from .adversary import BatchAdversaryHead, grad_reverse
from .film import FiLM
from .temperature_annealer import TemperatureAnnealer


__all__ = [
    # MLP modules
    'MLP',
    'ConditionalMLP',

    # GNN modules
    'init_gnn_module',
    'create_dynamic_gnn_module_class',

    # Vector Quantization
    'CosineSimCodebook',
    'get_vq_class',
    'get_valid_params',
    'ResidualVQ_Squint',
    'ConditionalVQ',

    # NB likelihood
    'DispersionHead',

    # Adversarial batch invariance (Ganin et al. 2015 GRL)
    'BatchAdversaryHead',
    'grad_reverse',

    # Conditioning modules
    'FiLM',

    # Annealing modules
    'TemperatureAnnealer',

]
