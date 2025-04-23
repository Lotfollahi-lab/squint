from typing import Callable, Union, Literal, List
from einops import rearrange

import torch
import pytorch_lightning as pl

from ..modules.cosine_codebook import CosineSimCodebook
from ..modules.gnn import init_gnn_module

class VQGraph_Encoder(pl.LightningModule):

    def __init__(
            self,
            num_linear_layers: int = 1,
            gnn_layer_name: str = 'SAGE',
            in_channels: int = None,
            hidden_channels: List[int] | int = 500,
            num_gnn_layers: int = 2,
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
        """
        Initialize the VQGraph_Encoder.

        Parameters
        ----------
        - num_linear_layers: int
            The number of linear layers to use before the GNN module.
        - gnn_layer_name: str
            The name of the GNN layer to use.
        - in_channels: int
            The input dimension of the GNN module.
        - hidden_channels: List[int] | int
            The hidden dimension of the GNN module.
        - num_gnn_layers: int
            The number of layers in the GNN module.
        - act_first: bool
            Whether to apply the activation function before the normalization layer.
        - activation: Union[str, Callable, None]
            The activation function to apply.
        - dropout: float
            The dropout rate.
        - norm: Union[str, Callable, None]
            The normalization layer to apply.
        - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
            The initialization method to use.
        - learnable_codebook: bool
            Whether to learn the codebook.
        - num_codebooks: int
            The number of codebooks to use.
        - codebook_size: int
            The size of the codebook.
        - decay: float
            The decay rate for the codebook.
        - eps: float
            The epsilon value for the codebook.
        - kmeans_init: bool
            Whether to use kmeans initialization for the codebook.
        - kmeans_iters: int
            The number of kmeans iterations to use.
        - sync_kmeans: bool
            Whether to synchronize the kmeans initialization across all processes.
        - threshold_ema_dead_code: int
            The threshold for the ema dead code.
        - use_ddp: bool
            Whether to use distributed data parallel training.
        - sample_codebook_temp: float
            The temperature for the codebook sampling.
        """
        super().__init__()

        # initialize the pre-VQ Graph Convolution module
        self.gnn_module = init_gnn_module(
            gnn_name=gnn_layer_name,
            num_linear_layers=num_linear_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            act_first=act_first,
            activation=activation,
            dropout=dropout,
            norm=norm,
            init_method=init_method
        )

        # initialize codebook class
        print(f"Initializing {num_codebooks} Cosine codebook(s) with {codebook_size} codes of dimension {self.gnn_module.hidden_channels}.")
        self._codebook = CosineSimCodebook(
            dim=self.gnn_module.hidden_channels,
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
        self.hidden_channels = self.gnn_module.hidden_channels


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
        - h_gnn: torch.Tensor
            Forward (output) of the GNN module.
        - h_vq: torch.Tensor
            Quantized node embeddings.
        - indices: torch.Tensor
            The indices of the node embeddings mapped to codebook embeddings.
        - dist: torch.Tensor
            The distances between the node embeddings and the codebook embeddings.
        - codebook_embeddings: torch.Tensor
            The codebook embeddings.
        """
        # forward pass of the GNN module
        h_gnn = self.gnn_module(
            batch_x,
            batch_edge_index
        )

        # VQ-encode the node embeddings
        h_vq, \
        indices, \
        dist, \
        codebook_embeddings \
            = self._codebook(h_gnn)

        # if gumble sampling is used, straight-through estimator is turned off
        if self._codebook.sample_codebook_temp == 0.0:
            h_vq = h_gnn + (h_vq - h_gnn).detach()

        return h_gnn, \
            h_vq, \
            indices, \
            dist, \
            codebook_embeddings