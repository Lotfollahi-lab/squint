from typing import Literal, Optional

import torch
import pytorch_lightning as pl

from ..modules.mlp import MLP as MLP_Module
from ..modules.gnn import init_gnn_module


class VanillaGNN_Encoder(pl.LightningModule):

    def __init__(
            self,
            in_channels: int,
            gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv',
            mlp_params: Optional[dict] = None,
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
        - mlp_params: Optional[dict]
            Keyword arguments for the MLP module.
            Default: None. If None, the MLP module will not be used.
        - gnn_params: dict
            Keyword arguments for the GNN module.
            Default: {}. If empty, GNN module is initialized with default parameters.
        """
        super().__init__()

        # if mlp_params is not provided, the MLP module will not be used
        if mlp_params is None:
            self.mlp_module = None

            # the GNN module will use the input channels as the input channels
            gnn_in_channels = in_channels

            self.mlp_layers = 0
        else:
            self.mlp_module = MLP_Module(
                in_channels=in_channels,
                **mlp_params,
            )
            
            # the GNN module will use the output channels of the MLP module as the input channels
            gnn_in_channels = self.mlp_module.out_channels
            
            self.mlp_layers = self.mlp_module.num_layers

        # initialize the GNN module
        self.gnn_module = init_gnn_module(
            in_channels=gnn_in_channels,
            gnn_name=gnn_name,
            gnn_params=gnn_params,
        )
        self.gnn_layers = self.gnn_module.num_layers
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