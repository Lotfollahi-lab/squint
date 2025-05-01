from typing import Optional

import torch
import torch.nn.functional as F
# from torch_geometric.nn import MLP as MLP_Module
import pytorch_lightning as pl

from ..modules.mlp import MLP as MLP_Module


class MLPSoftmax(pl.LightningModule):
    def __init__(
            self,
            name: str = 'MLPSoftmax',
            in_channels: int = None,
            out_channels: int = None,
            use_xy_coordinates: bool = False,
            mlp_params: dict = {},
        ):
        """
        Initialize the LinearSoftmax decoder.

        Parameters
        ----------
        - name: str
            The name of the decoder.
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - use_xy_coordinates: bool
            Whether to concatenate the spatial coordinates of the nodes to the input tensor.
        - mlp_params: dict
            The parameters for the MLP.
        """
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_params = mlp_params
        self.use_xy_coordinates = use_xy_coordinates
        if use_xy_coordinates:
            if in_channels is None:
                self.mlp_params['hidden_channels'][0] += 2
            else:
                self.in_channels = in_channels + 2
        else:
            self.in_channels = in_channels
        self.mlp_module = MLP_Module(
                                in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                mlp_params=self.mlp_params,
                            )


    def forward(
            self,
            x: torch.Tensor,
            read_depth: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Parameters
        ----------
        - x: torch.Tensor
            The input tensor.
            Dimensions: (batch_size, in_channels)
        - read_depth: torch.Tensor
            The read depth tensor.
            Dimensions: (batch_size, 1)

        Returns:
        -------
        - torch.Tensor:
            Output of the MLP followed by a softmax and a multiplication with the read depth.
        """
        xhat = self.mlp_module(x)
        xhat = F.softmax(xhat, dim=-1)
        xhat = xhat * read_depth.unsqueeze(-1)
        return xhat
