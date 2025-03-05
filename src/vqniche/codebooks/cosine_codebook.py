"""
CosineSimCodebook class is a PyTorch Lightning Module that implements the cosine similarity codebook for the VQ-VAE model.

Reference: https://github.com/YangLing0818/VQGraph/blob/main/vq.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import pytorch_lightning as pl
from einops import einsum, rearrange

from ..utils.vqgraph_helpers import l2norm, \
    uniform_init, \
    kmeans, \
    gumbel_sample, \
    batched_embedding, \
    distributed, \
    sample_vectors_distributed, \
    batched_sample_vectors, \
    noop


class CosineSimCodebook(pl.LightningModule):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0.0,
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0), strict=False)):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, "... -> 1 ..."), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, "1 ... -> ...")

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x):
        input_is_2dim = x.ndim == 2

        if input_is_2dim:
            x = rearrange(x, "b d -> b 1 d")

        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, "h ... d -> h (...) d")
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)

        # Changing order of einsum arguments to match function requirements
        dist = einsum(flatten, embed, "h n d, h c d -> h n c")
        # dist = einsum("h n d, h c d -> h n c", flatten, embed)

        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        # print(embed_ind.shape)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        # print(embed_onehot.shape)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)

            self.cluster_size.data.lerp_(bins, 1 - self.decay)

            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)

            # Changing order of einsum arguments to match function requirements
            embed_sum = einsum(flatten, embed_onehot, "h n d, h n c -> h c d")
            # embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)

            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, "... -> ... 1")
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(rearrange(zero_mask, "... -> ... 1"), embed, embed_normalized)

            self.embed.data.lerp_(embed_normalized, 1 - self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind))

        if input_is_2dim:
            quantize = rearrange(quantize, "b 1 d -> b d")
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        return quantize, embed_ind, dist, self.embed