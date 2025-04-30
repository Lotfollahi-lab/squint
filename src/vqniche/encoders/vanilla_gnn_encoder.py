from typing import Literal

import torch
import pytorch_lightning as pl

from ..modules.mlp import MLP as MLP_Module
from ..modules.gnn import init_gnn_module

class VanillaGNN_Encoder(pl.LightningModule):

    def __init__(
            self,
            in_channels: int,
            gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv',
            mlp_params: dict = {},
            gnn_params: dict = {},
        ):
        """
        Initialize the VanillaGNN_Encoder.

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
        """
        super().__init__()

        self.mlp_module = MLP_Module(
            in_channels=in_channels,
            mlp_params=mlp_params,
        )
        if self.mlp_module is None:
            gnn_in_channels = in_channels
        else:
            gnn_in_channels = self.mlp_module.channel_list[-1]

        # initialize the GNN module
        self.gnn_module = init_gnn_module(
            in_channels=gnn_in_channels,
            gnn_name=gnn_name,
            gnn_params=gnn_params,
        )

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
        - h_latent: torch.Tensor
            Forward (output) of just the GNN module or MLP followed by the GNN module.
        """
        # forward pass of the MLP module
        if self.mlp_module is not None:
            h_mlp = self.mlp_module(batch_x)
        else:
            h_mlp = batch_x

        # forward pass of the GNN module
        h_latent = self.gnn_module(
            h_mlp,
            batch_edge_index
        )

        return h_latent