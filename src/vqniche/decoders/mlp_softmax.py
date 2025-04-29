from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP as MLP_Module
import pytorch_lightning as pl

# from ..modules.mlp import MLP_Module


class MLPSoftmax(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            name: str,
            num_layers: int,
            hidden_channels: Optional[int] = None,
            dropout: float = 0.0,
            act: str = "relu",
            norm: str = "batch_norm"
        ):
        """
        Initialize the LinearSoftmax decoder.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - name: str
            The name of the decoder.
        - num_layers: int
            The number of layers in the MLP.
        - hidden_channels: int
            The number of hidden channels.
        - dropout: float
            The dropout rate.
        - act: str
            The activation function.
        - norm: str
            The normalization method.
        """
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.norm = norm

        self.mlp_module = MLP_Module(
                                in_channels=in_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_layers=num_layers,
                                dropout=dropout,
                                act=act,
                                norm=norm,
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
