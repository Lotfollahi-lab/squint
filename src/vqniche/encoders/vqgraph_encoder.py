from typing import Callable, Union, Literal
from einops import rearrange

import torch
import torch.nn as nn
import pytorch_lightning as pl

from vqniche.codebooks.cosine_codebook import CosineSimCodebook
from ..gnn_modules.sage_conv import SAGEConv_Module
from ..attribute_decoders.linear_softmax import LinearSoftmax


class VQGraph_Encoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 256,
        graphconv_layer_name: str = 'SAGEConv',
        num_layers: int = 2,
        act_first: bool = True,
        activation: Union[str, Callable, None] = "relu",
        dropout: float = 0.5,
        norm: Union[str, Callable, None] = None,
        init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
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
        super().__init__()

        # graph convolution
        self.graphconv_layer_name = graphconv_layer_name

        # initialize the pre-VQ Graph Convolution module
        print(f"Initializing the pre-VQ {self.graphconv_layer_name} module from {in_channels} to {hidden_channels} across {num_layers - 1} layer(s).")
        self.pre_vq_conv_module = self._init_graph_conv_module(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act_first=act_first,
            activation=activation,
            dropout=dropout,
            norm=norm,
            init_method=init_method
        )

        # initialize codebook class
        print(f"Initializing {num_codebooks} Cosine codebook(s) with {codebook_size} codes of dimension {hidden_channels}.")
        self._codebook = CosineSimCodebook(
            dim=hidden_channels,
            learnable_codebook=learnable_codebook,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            decay=decay,
            eps=eps,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters, # 10
            sync_kmeans=sync_kmeans, # True
            threshold_ema_dead_code=threshold_ema_dead_code, # 0
            use_ddp=use_ddp, # False
            sample_codebook_temp=sample_codebook_temp, # 0.0
        )


    def _init_graph_conv_module(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            act_first: bool,
            activation: Union[str, Callable, None],
            dropout: float,
            norm: Union[str, Callable, None],
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform'
        ) -> SAGEConv_Module:
        """
        Initialize the Graph Convolution module(s).

        This function sets the hidden channels and the number of layers for the Graph Convolution module(s). Then it initializes and returns the graph convolution module. Currently, only GraphSAGE is supported.

        If the vq_location is 'feature-space', the hidden channels are set to the in_channels and the number of layers is set to num_layers - 1. If the vq_location is 'latent-space', the hidden channels are set to the hidden_channels and the number of layers is set to num_layers.

        Parameters
        ----------
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

        Returns
        -------
        - SAGE_Encoder: torch.nn.Module
            The Graph Convolution module.
        """
        if self.graphconv_layer_name == 'SAGEConv':
            print(f"Graph Convolution applied from {in_channels} to {hidden_channels} across {num_layers} layer(s).")
            # initialize and return the graph convolution module
            return SAGEConv_Module(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            act_first=act_first,
                            activation=activation,
                            dropout=dropout,
                            norm=norm,
                            init_method=init_method
                        )
        else:
            raise ValueError(f"Graph convolution layer {self.graphconv_layer_name} not supported.")


    @property
    def codebook(self) -> torch.Tensor:
        """
        Retrieves the codebook embeddings from the codebook class.

        Returns
        -------
        - codebook_embeddings: torch.Tensor
            The codebook embeddings retrieved from the codebook class.
        """
        codebook_embeddings = self._codebook.embed
        return rearrange(codebook_embeddings, "1 ... -> ...")


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the VQGraph_Encoder model.

        Parameters:
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.

        Returns
        -------
        - h_pre_vq_conv: torch.Tensor
            Forward (output) of the pre-VQ Graph Convolution module.
        - h_vq: torch.Tensor
            VQ-encoded node embeddings.
        - indices: torch.Tensor
            The indices of the node embeddings mapped to codebook embeddings.
        - dist: torch.Tensor
            The distances between the node embeddings and the codebook embeddings.
        - codebook_embeddings: torch.Tensor
            The codebook embeddings.
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

        # if gumble sampling is used, straight-through estimator is turned off
        if self._codebook.sample_codebook_temp == 0.0:
            h_vq = h_pre_vq_conv + (h_vq - h_pre_vq_conv).detach()

        return h_pre_vq_conv, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings