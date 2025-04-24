from typing import Literal

import torch
import pytorch_lightning as pl

from ..modules.gnn import init_gnn_module
from ..modules.mlp import MLP_Module


class VanillaGNN_Encoder(pl.LightningModule):

    def __init__(
            self,
            in_channels: int,
            mlp_params: dict = {},
            gnn_params: dict = {},
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
        ):
        """
        Initialize the VanillaGNN_Encoder.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - mlp_params: dict
            Keyword arguments for the MLP module.
        - gnn_params: dict
            Keyword arguments for the GNN module.
        - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
            The initialization method to use.
        """
        super().__init__()

        # initialize the MLP module if num_layers > 0
        self.mlp_module = MLP_Module(
            in_channels=in_channels,
            **mlp_params,
            init_method=init_method,
        )
        if self.mlp_module is not None:
            gnn_in_channels = self.mlp_module.dim
        else:
            gnn_in_channels = in_channels

        # initialize the Vanilla GNN module
        self.gnn_module = init_gnn_module(
            in_channels=gnn_in_channels,
            **gnn_params,
            init_method=init_method
        )
        assert self.gnn_module is not None, "Number of GNN layers is 0. Please set num_layers to a positive integer."

        self.dim = self.gnn_module.dim


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

        return h_gnn