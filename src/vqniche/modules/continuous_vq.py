"""
ContinuousVQ — an identity / passthrough "quantizer" used to ablate the
discrete codebook bottleneck of the dual VQ-VAE.

It returns the encoder's continuous latent UNCHANGED in place of the
quantized vector, contributes ZERO quantization loss, and emits
placeholder indices. Swapping ``ResidualVQ_Squint`` for ``ContinuousVQ``
(via ``vq_name: "ContinuousVQ"`` in ``vq_cell_params`` / ``vq_niche_params``)
yields a model with an IDENTICAL encoder/decoder architecture and latent
dimensionality, but a CONTINUOUS (non-discretized) bottleneck. That is the
direct comparison for the WABI rebuttal question "does discretization, not
encoder/decoder capacity or depth, drive SQUINT's benchmark gain?".

Concretely, with ContinuousVQ in both branches:
  * ``z_q = z`` — the continuous encoder latent flows straight to the
    decoder (no nearest-codebook snap);
  * the commitment loss ``mse_commit_loss = MSE(z, z_q) == 0``;
  * the codebook-diversity term is dropped (there is no codebook).
Everything else (encoder, GNN, decoders, graph, sampler, NB recon,
adjacency BCE, within-batch contrastive cell loss) is untouched.

Drop-in contract (mirrors ``ResidualVQ_Squint.forward``):
  ``forward(z) -> (z_q, indices, loss)``
    z_q     : (B, D)  == z, unchanged (continuous passthrough)
    indices : (B, 1)  placeholder zeros (no discrete codes exist)
    loss    : (1,)    zeros (no quantization penalty)

It exposes the attributes the dual model / encoder read off a VQ module
(``codebook_size``, ``num_quantizers``, ``heads``,
``separate_codebook_per_head``) so logging and the inference-cache
metadata stay valid. It deliberately exposes NO ``_codebook`` / ``layers``
attribute — the dual model's codebook-embedding extraction is guarded to
fall back to ``None``, so codebook-only losses (orthogonality / diversity)
simply do not apply to a continuous model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousVQ(nn.Module):
    """Identity passthrough bottleneck (no vector quantization)."""

    def __init__(self, dim: int = None, **kwargs):
        super().__init__()
        # `dim` is supplied by the encoder (= the latent dimensionality of
        # the branch). Extra VQ kwargs (codebook sizes, diversity weights,
        # EMA settings, ...) are accepted and ignored — there is no codebook.
        self.dim = dim
        # Placeholder metadata so the dual model's startup print and its
        # inference-cache metadata (codebook_size / num_quantizers / heads /
        # separate_codebook_per_head) have valid values even though no real
        # codebook exists. A single degenerate "code" (index 0) is reported.
        self.codebook_size = 1
        self.num_quantizers = 1
        self.heads = 1
        self.separate_codebook_per_head = False

    def forward(self, z: torch.Tensor):
        """Continuous passthrough: the "quantized" output IS the input."""
        indices = torch.zeros(z.shape[0], 1, dtype=torch.long, device=z.device)
        loss = z.new_zeros(1)
        return z, indices, loss
