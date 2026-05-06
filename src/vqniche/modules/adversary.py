"""
Domain-adversarial batch classifier with gradient reversal.

Used by VQNiche_Dual as the missing batch-invariance pressure on the
encoder (the analog of NicheCompass's KL term over the Gaussian latent).
Mechanics:

    z_mlp ─┐                   gradient flow:
           │                       (encoder)
       gradient                       │  ∂L_adv / ∂z_mlp  =  -alpha * ∂CE / ∂z_mlp
       reversal       (forward)       │
       layer          (identity)      │
           │                          │
           ▼                          │
       BatchClassifier ── softmax ── CE(labels)

Forward: identity (no effect on activations).
Backward: gradient is multiplied by `-alpha`. Encoder receives the
*reversed* gradient, so its parameters are nudged in the direction
that makes z_mlp LESS predictive of batch. The classifier's own
parameters get the unmodified gradient (because the GRL only flips
sign for upstream params — the classifier itself sits AFTER the GRL).

Reference: Ganin, Y. & Lempitsky, V. (2015). Unsupervised Domain
Adaptation by Backpropagation. Standard recipe for batch correction
in scRNA-seq when a KL prior is unavailable (e.g. for VQ-VAE).
"""

from typing import List, Optional

import torch
import torch.nn as nn


class _GradientReversalFn(torch.autograd.Function):
    """Forward identity; backward negates the gradient (scaled by alpha)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # The second return is the gradient w.r.t. `alpha`, which is a
        # non-tensor scalar — return None.
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Gradient-reversal layer entry point. Use as `grad_reverse(z, alpha=...)`.
    `alpha` controls the strength of the adversarial signal: larger alpha
    means stronger pressure on the encoder to be batch-invariant. A
    typical schedule ramps alpha linearly from 0 to ~1 over training.
    """
    return _GradientReversalFn.apply(x, alpha)


class BatchAdversaryHead(nn.Module):
    """
    Small MLP classifier on `z_mlp` that predicts the per-cell batch label.

    Forward path applies gradient reversal *before* the classifier so
    backward gradient on upstream params (i.e. the encoder) is negated.
    The classifier's own params train normally (they're after the GRL).
    """

    def __init__(
            self,
            in_channels: int,
            n_batches: int,
            hidden_channels: Optional[List[int]] = None,
            dropout: float = 0.0,
        ):
        super().__init__()
        widths = list(hidden_channels) if hidden_channels else []
        layers: List[nn.Module] = []
        prev = in_channels
        for h in widths:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, n_batches))
        self.classifier = nn.Sequential(*layers)
        self.n_batches = int(n_batches)

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        z: (B, in_channels)
        Returns: (B, n_batches) batch logits.
        """
        z_rev = grad_reverse(z, alpha=alpha)
        return self.classifier(z_rev)
