from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LinearSoftmax(pl.LightningModule):
    def __init__(
            self,
            name: str,
            in_channels: int,
            out_channels: int
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
        """
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels)


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
