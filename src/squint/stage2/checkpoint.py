"""
Save / load a trained stage-2 model (torch).

Self-describing: writes the model weights next to a JSON of the full
``Stage2Config`` so a checkpoint can be reloaded without re-specifying the
architecture. Avoids PL's ``load_from_checkpoint`` (whose constructor takes a
dataclass, not the saved hyperparameter dict).
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import torch

from .config import Stage2Config
from .model import SpatialCodeTransformer

CONFIG_NAME = "stage2_config.json"
WEIGHTS_NAME = "stage2_model.pt"


def save_stage2(out_dir: str, model: SpatialCodeTransformer, cfg: Stage2Config) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, WEIGHTS_NAME))
    return out_dir


def load_stage2(
    out_dir: str, map_location: str = "cpu"
) -> Tuple[SpatialCodeTransformer, Stage2Config]:
    with open(os.path.join(out_dir, CONFIG_NAME)) as f:
        cfg = Stage2Config.from_dict(json.load(f))
    model = SpatialCodeTransformer(cfg)
    state = torch.load(os.path.join(out_dir, WEIGHTS_NAME), map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model, cfg
