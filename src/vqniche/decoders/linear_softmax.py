from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
import pytorch_lightning as pl


class LinearSoftmax(pl.LightningModule):
    def __init__(
            self,
            name: str,
            in_channels: int,
            out_channels: int,
            init_method: str = 'kaiming_uniform'
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
        - init_method: str
            The initialization method for the linear layer.
        """
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = Linear(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        weight_initializer=init_method,
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
            The output tensor after linear transformation and softmax.
        """
        x = self.linear(x)
        if self.name == 'LinearSoftmax':
            x = F.softmax(x, dim=-1)
            x = x * read_depth.unsqueeze(-1)
        return x
