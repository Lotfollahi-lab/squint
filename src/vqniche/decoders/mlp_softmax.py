import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from vqniche.modules.mlp import MLP as MLP_Module


class MLPSoftmax(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mlp_params: dict = {},
        ):
        """
        Initialize the LinearSoftmax decoder.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - mlp_params: dict
            The parameters for the MLP.
        """
        super().__init__()

        # set parameters of the MLPSoftmax decoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_params = mlp_params

        # initialize the MLP module
        self.mlp_module = MLP_Module(
                                in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                mlp_params=self.mlp_params,
                            )


    def forward(
            self,
            x: torch.Tensor,
            read_depth: torch.Tensor
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
