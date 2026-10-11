"""
LightningModule for training the stage-2 spatial code prior.

Wraps ``SpatialCodeTransformer``. Training minimises masked cross-entropy
(teacher-forced); validation additionally reports the *decoded* in-painting
accuracy (no teacher forcing -- the model fills the masked region with its own
iterative decoder), which is the metric that matches the downstream task.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import pytorch_lightning as pl

from .config import Stage2Config
from .model import SpatialCodeTransformer
from .decode import decode_patch


class Stage2LightningModule(pl.LightningModule):
    def __init__(self, cfg: Stage2Config):
        super().__init__()
        self.cfg = cfg
        self.model = SpatialCodeTransformer(cfg)
        # store a plain dict so checkpoints are self-describing / reloadable
        self.save_hyperparameters({"stage2_cfg": _cfg_to_dict(cfg)})

    # --------------------------------------------------------------- training
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self.model.loss(
            codes=batch["codes"],
            coords=batch["coords"],
            mask=batch["mask"],
            key_padding_mask=batch["key_padding_mask"],
            label_smoothing=self.cfg.optim.label_smoothing,
            l0_weight=self.cfg.optim.l0_weight,
        )
        self.log("train/loss", out["loss"], prog_bar=True, batch_size=batch["codes"].shape[0])
        for (b, l) in self.model.targets:
            key = f"{b}__{l}"
            self.log(f"train/acc_{key}", out[f"acc_{key}"], batch_size=batch["codes"].shape[0])
        return out["loss"]

    # ------------------------------------------------------------- validation
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        bs = batch["codes"].shape[0]
        # (a) teacher-forced masked CE / accuracy
        out = self.model.loss(
            codes=batch["codes"],
            coords=batch["coords"],
            mask=batch["mask"],
            key_padding_mask=batch["key_padding_mask"],
        )
        self.log("val/loss", out["loss"], prog_bar=True, batch_size=bs)
        for (b, l) in self.model.targets:
            self.log(f"val/acc_tf_{b}__{l}", out[f"acc_{b}__{l}"], batch_size=bs)

        # (b) decoded in-painting accuracy (no teacher forcing)
        filled = decode_patch(
            self.model,
            codes=batch["codes"],
            coords=batch["coords"],
            mask=batch["mask"],
            key_padding_mask=batch["key_padding_mask"],
            cfg=self.cfg.decode,
        )
        sup = batch["mask"] & (~batch["key_padding_mask"])
        if sup.any():
            for ti, (b, l) in enumerate(self.model.targets):
                correct = (filled[:, :, ti] == batch["codes"][:, :, ti]) & sup
                acc = correct.sum().float() / sup.sum().float()
                self.log(f"val/acc_dec_{b}__{l}", acc, prog_bar=(ti == 0), batch_size=bs)
        return out["loss"]

    # ------------------------------------------------------------- optimisers
    def configure_optimizers(self):
        oc = self.cfg.optim
        opt = torch.optim.AdamW(
            self.parameters(), lr=oc.lr, betas=oc.betas, weight_decay=oc.weight_decay
        )

        def lr_lambda(step: int) -> float:
            if step < oc.warmup_steps:
                return (step + 1) / max(1, oc.warmup_steps)
            prog = (step - oc.warmup_steps) / max(1, oc.max_steps - oc.warmup_steps)
            prog = min(1.0, max(0.0, prog))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    # ----------------------------------------------------------- grad clip cfg
    def on_before_optimizer_step(self, optimizer):
        if self.cfg.optim.grad_clip and self.cfg.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.optim.grad_clip)


def _cfg_to_dict(cfg: Stage2Config) -> dict:
    """Flatten the dataclass config to a checkpoint-friendly dict."""
    import dataclasses

    def conv(x):
        if dataclasses.is_dataclass(x):
            return {f.name: conv(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    return conv(cfg)
