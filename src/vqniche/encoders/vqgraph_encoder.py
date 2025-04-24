from typing import Literal
from einops import rearrange

import torch
import pytorch_lightning as pl

from ..modules.cosine_codebook import CosineSimCodebook
from ..modules.gnn import init_gnn_module
from ..modules.mlp import MLP_Module


class VQGraph_Encoder(pl.LightningModule):

    def __init__(
            self,
            in_channels: int = None,
            gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv',
            mlp_params: dict = {},
            gnn_params: dict = {},
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
            codebook_params: dict = {},
        ):
        """
        Initialize the VQGraph_Encoder.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv']
            The name of the GNN module.
        - mlp_params: dict
            Keyword arguments for the MLP module.
        - gnn_params: dict
            Keyword arguments for the GNN module.
        - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
            The initialization method to use.
        - codebook_params: dict
            Keyword arguments for the codebook.
        """
        super().__init__()

        # initialize the MLP module
        if mlp_params['num_layers'] == 0:
            self.mlp_module = None
            gnn_in_channels = in_channels
        else:
            self.mlp_module = MLP_Module(
                in_channels=in_channels,
                **mlp_params,
                init_method=init_method,
            )
            gnn_in_channels = self.mlp_module.dim

        # initialize the GNN module
        self.gnn_module = init_gnn_module(
            in_channels=gnn_in_channels,
            gnn_name=gnn_name,
            **gnn_params,
            init_method=init_method,
        )
        assert self.gnn_module is not None, "Number of GNN layers is 0. Please set num_layers to a positive integer."

        # initialize the codebook
        self._codebook = CosineSimCodebook(
            dim=self.gnn_module.dim,
            **codebook_params,
        )
        self.dim = self.gnn_module.dim


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
        # forward pass of the MLP module
        if self.mlp_module is not None:
            h_mlp = self.mlp_module(batch_x)
        else:
            h_mlp = batch_x

        # forward pass of the GNN module
        h_gnn = self.gnn_module(
            h_mlp,
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