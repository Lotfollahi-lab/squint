"""
Configuration for the SQUINT stage-2 spatial code prior.

Pure-python dataclasses (no torch / no numpy) so the config can be imported,
serialised and unit-tested without the heavy ML stack. Everything the model,
data module and decoder need to agree on lives here.

Design recap (see ``README.md``):

    Stage 1 = SQUINT VQ-VAE  (FROZEN)  -> discrete code stacks per cell
    Stage 2 = this module              -> a graph-native, MaskGIT-style
                                          masked-code transformer that predicts
                                          held-out cells' code stacks from the
                                          spatial context of observed cells.

The stage-2 model never sees expression. It consumes only:
  * the integer code stacks of OBSERVED cells,
  * the 2D positions of ALL cells (observed + held-out),
and predicts the code stacks of the held-out cells. Decoding those codes back
to expression is done by SQUINT's frozen Negative-Binomial decoder (optional,
post-hoc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Branch / codebook description
# ---------------------------------------------------------------------------
@dataclass
class BranchSpec:
    """Describes one residual code stack (a SQUINT branch).

    A SQUINT cell carries two such stacks: a ``cell`` (cell-type identity)
    stack and a ``niche`` (spatial domain) stack, each with ``num_levels``
    residual levels of sizes ``codebook_sizes`` (e.g. ``[30, 90]``).
    """

    name: str                       # "cell" or "niche"
    codebook_sizes: List[int]       # per-level vocabulary, e.g. [30, 90]

    # adata storage keys (match SQUINT's predict pipeline; see report_codebook_usage.py)
    uns_key: str = ""               # e.g. "Indices_cell"
    sizes_key: str = ""             # e.g. "codebook_sizes_cell"

    @property
    def num_levels(self) -> int:
        return len(self.codebook_sizes)

    def __post_init__(self) -> None:
        if not self.codebook_sizes:
            raise ValueError(f"branch {self.name!r}: codebook_sizes must be non-empty")
        if any(int(k) <= 0 for k in self.codebook_sizes):
            raise ValueError(f"branch {self.name!r}: codebook sizes must be positive")
        self.codebook_sizes = [int(k) for k in self.codebook_sizes]
        if not self.uns_key:
            self.uns_key = f"Indices_{self.name}"
        if not self.sizes_key:
            self.sizes_key = f"codebook_sizes_{self.name}"


def default_branches(
    codebook_sizes_cell: List[int],
    codebook_sizes_niche: List[int],
) -> List[BranchSpec]:
    """The standard SQUINT dual-branch layout: cell first, then niche.

    Order matters: the prediction order (and the hierarchical cell->niche
    conditioning) follows the order of this list.
    """
    return [
        BranchSpec("cell", list(codebook_sizes_cell)),
        BranchSpec("niche", list(codebook_sizes_niche)),
    ]


# ---------------------------------------------------------------------------
# Data / patching
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """How patches and masks are drawn from a frozen ``predicted_adata``."""

    coord_key: str = "spatial"          # adata.obsm key for 2D positions
    section_key: str = "adata_batch_id" # adata.obs key splitting tissue sections

    patch_size: int = 1024              # cells per training patch (spatial disk)
    knn: int = 16                       # graph degree used for patch growth / contiguity

    # contiguous-block masking (matches the region-holdout geometry)
    mask_kind: str = "block"            # "block" (contiguous) | "random" (scatter)
    mask_frac_min: float = 0.10         # fraction of patch cells held out (lo)
    mask_frac_max: float = 0.50         # fraction of patch cells held out (hi)

    # epoch sizing: patches drawn per epoch = oversample * (N_cells / patch_size)
    oversample: float = 1.0

    # coordinate normalisation per patch (centre + scale) before positional enc.
    # "std"   -> divide by per-patch coordinate std
    # "knn"   -> divide by median nearest-neighbour distance in the patch
    coord_norm: str = "std"

    def __post_init__(self) -> None:
        if self.patch_size < 4:
            raise ValueError("patch_size too small")
        if not (0.0 < self.mask_frac_min <= self.mask_frac_max < 1.0):
            raise ValueError("require 0 < mask_frac_min <= mask_frac_max < 1")
        if self.mask_kind not in ("block", "random"):
            raise ValueError("mask_kind must be 'block' or 'random'")
        if self.coord_norm not in ("std", "knn"):
            raise ValueError("coord_norm must be 'std' or 'knn'")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """The graph-native Contextual RQ-Transformer."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # 2D Fourier positional features (Random Fourier Features over coords)
    pos_num_freqs: int = 64
    pos_sigma: float = 1.0              # frequency scale (in normalised-coord units)
    pos_learnable_freqs: bool = False   # keep RFF matrix fixed by default (reproducible)

    # spatial attention bias: additive  -gamma_h * dist(i, j)  per head.
    # gamma is a learnable positive per-head scalar initialised at this value.
    attn_dist_bias: bool = True
    attn_gamma_init: float = 1.0

    # hierarchical decoding: niche heads condition on the (teacher-forced)
    # cell codes, mirroring SQUINT's FiLM cell->niche coupling.
    hierarchical: bool = True

    # tie the input code-embedding tables to the output-head conditioning
    # embeddings (weight sharing, MaskGIT-style).
    tie_code_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


# ---------------------------------------------------------------------------
# Inference / iterative decoding (MaskGIT)
# ---------------------------------------------------------------------------
@dataclass
class DecodeConfig:
    """Confidence-scheduled parallel decoding for in-painting."""

    steps: int = 12                     # number of unmask iterations
    schedule: str = "cosine"            # fraction-unmasked schedule
    temperature: float = 1.0            # sampling temperature (0 -> argmax)
    noise_anneal: float = 1.0           # Gumbel confidence-noise scale, annealed to 0
    context_radius_mult: float = 2.0    # observed context ring = this * masked-region radius

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.schedule not in ("cosine", "linear"):
            raise ValueError("schedule must be 'cosine' or 'linear'")


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    batch_size: int = 8                 # patches per step


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class Stage2Config:
    branches: List[BranchSpec]
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 0

    # ----- convenience -----------------------------------------------------
    @classmethod
    def squint_default(
        cls,
        codebook_sizes_cell: List[int] = (30, 90),
        codebook_sizes_niche: List[int] = (30, 90),
        **overrides,
    ) -> "Stage2Config":
        """The default dual-branch RVQ(30, 90) SQUINT layout."""
        return cls(
            branches=default_branches(
                list(codebook_sizes_cell), list(codebook_sizes_niche)
            ),
            **overrides,
        )

    @property
    def prediction_targets(self) -> List[Tuple[str, int]]:
        """Ordered (branch_name, level) prediction targets.

        The order defines both the residual-depth coarse->fine ordering within
        a branch and the cross-branch order (cell stack fully before niche
        stack), which is what the hierarchical conditioning consumes.
        """
        targets: List[Tuple[str, int]] = []
        for b in self.branches:
            for lvl in range(b.num_levels):
                targets.append((b.name, lvl))
        return targets

    def branch(self, name: str) -> BranchSpec:
        for b in self.branches:
            if b.name == name:
                return b
        raise KeyError(name)

    # ----- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Stage2Config":
        """Rebuild from ``to_dict`` output (e.g. a saved stage2_config.json)."""
        return cls(
            branches=[
                BranchSpec(
                    name=b["name"],
                    codebook_sizes=list(b["codebook_sizes"]),
                    uns_key=b.get("uns_key", ""),
                    sizes_key=b.get("sizes_key", ""),
                )
                for b in d["branches"]
            ],
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            decode=DecodeConfig(**d.get("decode", {})),
            optim=_optim_from_dict(d.get("optim", {})),
            seed=int(d.get("seed", 0)),
        )


def _optim_from_dict(d: dict) -> OptimConfig:
    d = dict(d)
    if "betas" in d and d["betas"] is not None:
        d["betas"] = tuple(d["betas"])      # json round-trips tuple -> list
    return OptimConfig(**d)
