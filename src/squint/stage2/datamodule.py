"""
Torch Dataset / collate / DataModule for the stage-2 prior.

Wraps the pure-numpy ``PatchSampler`` (data.py). Each item is one spatial patch;
``collate_patches`` pads a batch of variable-sized patches to a common length
with a key-padding mask, so the transformer attends only over real cells.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Stage2Config
from .data import AnnDataCodeSource, PatchSampler, Patch


def patch_to_arrays(
    patch: Patch, targets: Sequence[Tuple[str, int]]
) -> Dict[str, np.ndarray]:
    """Flatten a Patch into model-ready arrays in the target order."""
    P = patch.size
    T = len(targets)
    codes = np.zeros((P, T), dtype=np.int64)
    for ti, (b, l) in enumerate(targets):
        codes[:, ti] = patch.codes[b][:, l]
    return {
        "codes": codes,                          # (P, T)
        "coords": patch.coords_norm.astype(np.float32),  # (P, 2)
        "mask": patch.mask.astype(bool),         # (P,)
    }


class SpatialCodeDataset(Dataset):
    """Streams spatial patches from a frozen code source.

    Patches are i.i.d. spatial disks; ``idx`` indexes the epoch, not a fixed
    patch. With ``deterministic=True`` the patch for a given ``idx`` is
    reproducible (used by tests).
    """

    def __init__(
        self,
        source: AnnDataCodeSource,
        cfg: Stage2Config,
        length: Optional[int] = None,
        deterministic: bool = False,
        seed: int = 0,
        restrict_to_train: bool = False,
    ):
        self.source = source
        self.cfg = cfg
        self.sampler = PatchSampler(source, cfg.data, restrict_to_train=restrict_to_train)
        self.targets = cfg.prediction_targets
        self._length = int(length) if length is not None else self.sampler.epoch_len()
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return self._length

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng([self.seed, idx])
        wi = torch.utils.data.get_worker_info()
        worker = wi.id if wi is not None else 0
        salt = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        return np.random.default_rng([self.seed, worker, idx, salt])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        patch = self.sampler.sample(self._rng(idx))
        return patch_to_arrays(patch, self.targets)


def collate_patches(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Pad a list of patches to the max length; build a key-padding mask."""
    B = len(batch)
    P = max(item["codes"].shape[0] for item in batch)
    T = batch[0]["codes"].shape[1]

    codes = torch.zeros(B, P, T, dtype=torch.long)
    coords = torch.zeros(B, P, 2, dtype=torch.float32)
    mask = torch.zeros(B, P, dtype=torch.bool)
    key_padding_mask = torch.ones(B, P, dtype=torch.bool)   # True == PAD

    for i, item in enumerate(batch):
        p = item["codes"].shape[0]
        codes[i, :p] = torch.from_numpy(item["codes"])
        coords[i, :p] = torch.from_numpy(item["coords"])
        mask[i, :p] = torch.from_numpy(item["mask"])
        key_padding_mask[i, :p] = False                     # real cells
    return {
        "codes": codes,
        "coords": coords,
        "mask": mask,
        "key_padding_mask": key_padding_mask,
    }


class Stage2DataModule:
    """Minimal data module (framework-agnostic; works with PL Trainer too).

    Splits sections into train/val so the held-out region task is evaluated on
    sections the model trained on (in-section in-painting) by default; pass
    ``val_sections`` to hold entire sections out instead.
    """

    def __init__(
        self,
        source: AnnDataCodeSource,
        cfg: Stage2Config,
        num_workers: int = 4,
        val_fraction: float = 0.0,
        val_sections: Optional[Sequence[int]] = None,
        restrict_train_to_split: bool = False,
    ):
        self.source = source
        self.cfg = cfg
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.val_sections = val_sections
        # When the source carries a data-split mask, confine BOTH train and val
        # patches to non-held-out cells so neither training nor the val metric
        # ever touches the held-out region (which is the eval target).
        self.restrict_to_train = bool(restrict_train_to_split) and source.has_holdout

    def train_dataloader(self) -> DataLoader:
        # Patches are i.i.d. random samples, so the "epoch" is artificial and
        # total training is governed by max_steps. We only need each epoch to
        # contain enough patches to (a) never yield zero batches and (b) keep
        # per-epoch overhead low. Floor the length at a healthy multiple of the
        # batch size (the raw N/patch_size can be tiny for small sections).
        bs = self.cfg.optim.batch_size
        length = max(self.sampler_len(), 8 * bs, 64)
        ds = SpatialCodeDataset(self.source, self.cfg, length=length, seed=self.cfg.seed,
                                restrict_to_train=self.restrict_to_train)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_patches,
            drop_last=False,                # never drop the only (partial) batch
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        # validation patches: deterministic stream for stable metrics
        bs = self.cfg.optim.batch_size
        length = max(2 * bs, self.sampler_len() // 5, 16)
        ds = SpatialCodeDataset(
            self.source, self.cfg, length=length, deterministic=True, seed=self.cfg.seed + 1,
            restrict_to_train=self.restrict_to_train,
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_patches,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def sampler_len(self) -> int:
        return PatchSampler(
            self.source, self.cfg.data, restrict_to_train=self.restrict_to_train
        ).epoch_len()
