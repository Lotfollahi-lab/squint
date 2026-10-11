"""
Code-conditional dispersion head for the negative-binomial reconstruction loss.

Default behaviour (when this head is NOT instantiated): a single learnable
parameter `theta ∈ R^{n_genes}`, shared across all cells. This is the
classical scVI-VAE / NicheCompass setup.

When this head IS instantiated: a small MLP maps the quantized latent
embedding `z_q ∈ R^{B × D}` of each cell to a per-cell, per-gene log-
dispersion `log(theta) ∈ R^{B × n_genes}`. The dispersion thus depends on
the assigned VQ code (via `z_q`), letting the model say "in niche A this
gene is tight, in niche B it is noisy" — a strictly more expressive NB
likelihood.

We condition on `z_q` (continuous quantized embedding) rather than on the
discrete code id because (a) it is variant-agnostic — works identically for
single-codebook VQ, RVQ, ConditionalVQ, and multi-head VQ — and (b) it is
the form used by scVI-tools.

Shapes:
    Input:  h_quantized of shape (B, D), where D is the encoder/codebook dim.
    Output: dispersion (= exp(log_theta) + eps) of shape (B, n_genes).

The output is broadcastable against the (B, n_genes) NB target tensors,
so the existing `nb_attribute_reconstruction_loss` accepts it without
modification (the loss only relies on broadcasting between `dispersion`,
`pred_attr`, and `target_attr`).
"""

from typing import List, Optional

import torch
import torch.nn as nn


class DispersionHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: Optional[List[int]] = None,
            activation: str = "relu",
            min_log_dispersion: float = -8.0,
            max_log_dispersion: float =  8.0,
        ):
        """
        Parameters
        ----------
        in_channels: int
            Dimensionality of the quantized latent (encoder/codebook dim).
        out_channels: int
            Number of genes (per-gene dispersion).
        hidden_channels: list of int, optional
            Hidden layer widths. None or empty list -> a single linear layer
            (no hidden activations). Default: None (linear).
        activation: str
            One of "relu", "gelu". Used between hidden layers only.
        min_log_dispersion, max_log_dispersion: float
            Clamp range for the predicted log-dispersion before exp(). Keeps
            the NB likelihood numerically stable when the head sees out-of-
            distribution latent points.
        """
        super().__init__()
        self.min_log_dispersion = float(min_log_dispersion)
        self.max_log_dispersion = float(max_log_dispersion)

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU}.get(activation, nn.ReLU)
        layers: List[nn.Module] = []
        widths = list(hidden_channels) if hidden_channels else []
        prev = in_channels
        for h in widths:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, out_channels))
        self.net = nn.Sequential(*layers)

        # Initialize the final layer's bias to 0 so that exp(log_theta) starts
        # at ~1, matching the magnitude of the legacy randn-initialised
        # `BaseModel.dispersion`. Avoids exploding dispersion at step 0.
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, h_quantized: torch.Tensor) -> torch.Tensor:
        """
        h_quantized: (B, D)
        Returns: dispersion (B, n_genes), strictly positive.
        """
        log_theta = self.net(h_quantized)
        log_theta = log_theta.clamp(
            min=self.min_log_dispersion, max=self.max_log_dispersion
        )
        # eps is added inside the NB loss already, so no extra eps needed here
        return torch.exp(log_theta)
