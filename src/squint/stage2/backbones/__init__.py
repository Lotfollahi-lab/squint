"""
Swappable stage-2 backbones (selected by ``config.model.arch``).

All backbones share the (x, coords, dist, key_padding_mask, mask) -> h contract
documented in ``base.Backbone`` — so the embedding, prediction heads, masked-CE
loss and MaskGIT decode are identical across architectures and the archs form a
clean head-to-head ablation.

Imports are LAZY (only the selected arch's module is imported), so a backbone
that needs an optional dependency can't break package import for the others,
and the numpy-only core still imports without torch.
"""

from __future__ import annotations


def build_backbone(cfg):
    """Construct the backbone for ``cfg.model.arch``."""
    arch = cfg.model.arch
    if arch == "transformer":
        from .transformer_backbone import TransformerBackbone
        return TransformerBackbone(cfg)
    if arch == "gnn":
        from .gnn import GNNBackbone
        return GNNBackbone(cfg)
    if arch == "labelprop":
        from .labelprop import LabelPropBackbone
        return LabelPropBackbone(cfg)
    if arch == "graphmae":
        from .graphmae import GraphMAEBackbone
        return GraphMAEBackbone(cfg)
    if arch == "gps":
        from .gps import GPSBackbone
        return GPSBackbone(cfg)
    if arch == "diffusion":
        from .diffusion import DiffusionBackbone
        return DiffusionBackbone(cfg)
    raise ValueError(f"unknown stage-2 arch {arch!r}")
