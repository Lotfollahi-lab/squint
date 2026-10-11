"""
SQUINT stage-2 spatial code prior.

A graph-native, MaskGIT-style masked-code transformer that predicts held-out
cells' discrete code stacks from the spatial context of observed cells. It sits
*on top of* a FROZEN SQUINT VQ-VAE (stage 1) and never touches the stage-1 code
or training path -- it consumes only exported codes + positions.

Public API
----------
Config / data core (no torch needed):
    Stage2Config, BranchSpec, DataConfig, ModelConfig, DecodeConfig, OptimConfig
    AnnDataCodeSource, PatchSampler, Patch, inpainting_patch

Model / training / decoding (require torch / pytorch_lightning -- imported lazily):
    SpatialCodeTransformer
    Stage2LightningModule, Stage2DataModule, SpatialCodeDataset, collate_patches
    inpaint

The torch-dependent symbols are loaded lazily via ``__getattr__`` so that the
pure-numpy data/masking/config layer can be imported and unit-tested without
torch installed.
"""

from __future__ import annotations

# ---- light (numpy-only) re-exports --------------------------------------
from .config import (
    Stage2Config,
    BranchSpec,
    DataConfig,
    ModelConfig,
    DecodeConfig,
    OptimConfig,
    default_branches,
)
from .data import (
    AnnDataCodeSource,
    PatchSampler,
    Patch,
    inpainting_patch,
    knn_indices,
    normalise_coords,
)
from . import masking

# ---- heavy (torch) re-exports, loaded on demand -------------------------
_LAZY = {
    "SpatialCodeTransformer": ("model", "SpatialCodeTransformer"),
    "Stage2LightningModule": ("lightning", "Stage2LightningModule"),
    "Stage2DataModule": ("datamodule", "Stage2DataModule"),
    "SpatialCodeDataset": ("datamodule", "SpatialCodeDataset"),
    "collate_patches": ("datamodule", "collate_patches"),
    "inpaint": ("decode", "inpaint"),
    "decode_patch": ("decode", "decode_patch"),
    "region_holdout_eval": ("evaluate", "region_holdout_eval"),
    "save_stage2": ("checkpoint", "save_stage2"),
    "load_stage2": ("checkpoint", "load_stage2"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Stage2Config", "BranchSpec", "DataConfig", "ModelConfig", "DecodeConfig",
    "OptimConfig", "default_branches",
    "AnnDataCodeSource", "PatchSampler", "Patch", "inpainting_patch",
    "knn_indices", "normalise_coords", "masking",
    *_LAZY.keys(),
]
