import torch
import torch.nn as nn
import torch.functional as F
from torch.cuda.amp import autocast

import pytorch_lightning as pl

from torch_geometric.nn import GraphSAGE as SAGE_Encoder

from typing import List, Tuple, Callable, Optional, Union
from einops import einsum, rearrange, pack, repeat, unpack

from vqniche.utils.vqgraph_helpers import *


class VQGraph_Encoder(pl.LightningModule):

    def __init__(
        self,
        graphconv_layer_name: str = 'SAGEConv',
        in_channels: int = None,
        hidden_channels: int = 256,
        num_layers: int = 2,
        act_first: bool = True,
        activation: Union[str, Callable, None] = "relu",
        dropout: float = 0.5,
        norm: Union[str, Callable, None] = None,
        learnable_codebook: bool = True,
        num_codebooks: int = 1,
        codebook_size: int = 256,
        decay: float = 0.8,
        eps: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sync_kmeans: bool = True,
        threshold_ema_dead_code: int = 0,
        use_ddp: bool = False,
        sample_codebook_temp: float = 0.0,
    ):
        super(VQGraph_Encoder, self).__init__()

        # graph convolution
        self.graphconv_layer_name = graphconv_layer_name

        # initialize the pre-VQ Graph Convolution module
        print("Initializing the pre-VQ Graph Convolution module.")

        self.pre_vq_conv_module = self._init_graph_conv_module(
            in_channels=in_channels,
            hidden_channels=in_channels,
            num_layers=num_layers - 1,
            act_first=act_first,
            activation=activation,
            dropout=dropout,
            norm=norm
        )

        # initialize codebook class
        self._codebook = CosineSimCodebook(
            dim=in_channels,
            learnable_codebook=learnable_codebook, # True
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            decay=decay,
            eps=eps,
            kmeans_init=kmeans_init, # False
            kmeans_iters=kmeans_iters, # 10
            sync_kmeans=sync_kmeans, # True
            threshold_ema_dead_code=threshold_ema_dead_code, # 0
            use_ddp=use_ddp, # False
            sample_codebook_temp=sample_codebook_temp, # 0.0
        )

        # initialize the decoder module for the node attributes
        print("Initializing the decoder module for node attributes.")
        self.decoder_node = nn.Linear(
                                in_features=in_channels,
                                out_features=in_channels
                            )

        # initialize the decoder module for the adjacency matrix
        print("Initializing the decoder module for the adjacency matrix.")
        self.decoder_edge = nn.Linear(
                                in_features=in_channels,
                                out_features=in_channels
                            )

        # initialize the post-VQ Graph Convolution module
        print("Initializing the post-VQ Graph Convolution module.")
        self.post_vq_conv_module = self._init_graph_conv_module(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=1,
            act_first=act_first,
            activation=activation,
            dropout=0.0,
            norm=None
        )


    def _init_graph_conv_module(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            act_first: bool,
            activation: Union[str, Callable, None],
            dropout: float,
            norm: Union[str, Callable, None]
        ) -> SAGE_Encoder:
        """
        Initialize the Graph Convolution module(s).

        This function sets the hidden channels and the number of layers for the Graph Convolution module(s). Then it initializes and returns the graph convolution module. Currently, only GraphSAGE is supported.

        If the vq_location is 'feature-space', the hidden channels are set to the in_channels and the number of layers is set to num_layers - 1. If the vq_location is 'latent-space', the hidden channels are set to the hidden_channels and the number of layers is set to num_layers.

        Args:
            - in_channels: int
                The input dimension of the Graph Convolution module.
            - hidden_channels: int
                The hidden dimension of the Graph Convolution module.
            - num_layers: int
                The number of layers in the pre-VQ Graph Convolution module.
            - act_first: bool
                Whether to apply the activation function before the normalization layer.
            - activation: Union[str, Callable, None]
                The activation function to apply.
            - dropout: float
                The dropout rate.
            - norm: Union[str, Callable, None]
                The normalization layer to apply.

        Returns:
            - SAGE_Encoder: The Graph Convolution module.
        """
        if self.graphconv_layer_name == 'SAGEConv':
            print(f"Graph Convolution applied from {in_channels} to {hidden_channels} across {num_layers} layer(s).")
            # initialize and return the graph convolution module
            return SAGE_Encoder(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            act_first=act_first,
                            act=activation,
                            dropout=dropout,
                            norm=norm
                        )
        else:
            raise ValueError(f"Graph convolution layer {self.graphconv_layer_name} not supported.")


    @property
    def codebook(self) -> torch.Tensor:
        """
        Retrieves the codebook embeddingsfrom the codebook class.

        Returns:
            - codebook_embeddings: torch.Tensor
                The codebook embeddings retrieved from the codebook class.
        """
        # NOTE: This function is not used in the forward pass.
        codebook_embeddings = self._codebook.embed
        return rearrange(codebook_embeddings, "1 ... -> ...")


    def get_codes_from_indices(
            self,
            indices: torch.Tensor
        ) -> torch.Tensor:
        """
        Retrieves the codes from the codebook class based on the provided indices.

        Args:
            - indices: torch.Tensor
                The indices of the codes to retrieve.

        Returns:
            - codes: torch.Tensor
                The codes retrieved from the codebook class.
        """
        # NOTE: This function is not used in the forward pass.
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack([indices], "b * h")
        indices = rearrange(indices, "b n h -> b h n")

        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        (codes,) = unpack(codes, ps, "b * d")
        return codes


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the VQGraph_Encoder model.
        """
        # pre-VQ Graph Convolution
        h_pre_vq_conv = self.pre_vq_conv_module(
            batch_x,
            batch_edge_index
        )

        # VQ-encode the node embeddings
        h_vq, \
        indices, \
        dist, \
        codebook_embeddings \
            = self._codebook(h_pre_vq_conv)

        # decode the VQ-encoded node embeddings to recover the node attributes
        h_node = self.decoder_node(h_vq)

        # decode the VQ-encoded edge embeddings to recover the adjacency matrix
        h_edge = self.decoder_edge(h_vq)

        # post-VQ Graph Convolution
        h_post_vq_conv = self.post_vq_conv_module(
                            h_edge,
                            batch_edge_index
                        )

        return h_pre_vq_conv, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings, \
            h_node, \
            h_edge, \
            h_post_vq_conv


class VectorQuantize(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.8,
        eps: float = 1e-5,
        learnable_codebook: bool = False,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sync_kmeans: bool = True,
        threshold_ema_dead_code: int = 0,
        sync_codebook: bool = False,
        sample_codebook_temp: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        self._codebook = CosineSimCodebook(
            dim=self.dim,
            num_codebooks=1,
            codebook_size=self.codebook_size,
            decay=self.decay,
            eps=self.eps,
            learnable_codebook=learnable_codebook, # False
            kmeans_init=kmeans_init, # False
            kmeans_iters=kmeans_iters, # 10
            sync_kmeans=sync_kmeans, # True
            threshold_ema_dead_code=threshold_ema_dead_code, # 0
            use_ddp=sync_codebook, # False
            sample_codebook_temp=sample_codebook_temp, # 0.0
        )


    @property
    def codebook(self):
        codebook = self._codebook.embed
        return rearrange(codebook, "1 ... -> ...")

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack([indices], "b * h")
        indices = rearrange(indices, "b n h -> b h n")

        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        (codes,) = unpack(codes, ps, "b * d")
        return codes

    def forward(self, x):
        only_one = x.ndim == 2

        if only_one:
            x = rearrange(x, "b d -> b 1 d")

        quantize, embed_ind, dist, embed = self._codebook(x)

        codes = self.get_codes_from_indices(embed_ind)

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        return quantize, embed_ind, dist, self._codebook.embed


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
        if x.ndim == 2:
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

        if x.ndim == 2:
            quantize = rearrange(quantize, "b 1 d -> b d")
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        return quantize, embed_ind, dist, self.embed