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
            The number of input channels representing the number of dimensions of the latent embeddings from the encoder.
        - out_channels: int
            The number of output channels representing the number of dimensions of the input features to be reconstructed.
        
        - mlp_params: dict
            MLP-related hyperparameters such as `hidden_channels` (number of hidden channels representing the number of dimensions of the hidden features in the intermediate layers of the MLP), `dropout` (dropout rate), `act` (activation function), and `norm` (normalization function).
        """
        super().__init__()

        # set parameters of the MLPSoftmax decoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.mlp_module = MLP_Module(
            in_channels=in_channels,
            out_channels=out_channels,
            **mlp_params,
            plain_last=True,
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
