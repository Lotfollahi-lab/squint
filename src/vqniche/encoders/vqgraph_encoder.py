import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GraphSAGE as SAGE_Encoder

from typing import Callable, Union
from einops import rearrange, pack, repeat, unpack

from vqniche.codebooks.cosine_codebook import CosineSimCodebook

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