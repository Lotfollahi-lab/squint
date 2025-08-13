from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from vqniche.modules.mlp import MLP as MLP_Module
from vqniche.modules.mlp import ConditionalMLP as ConditionalMLP_Module
from vqniche.modules.film import FiLM
from .temperature_annealer import TemperatureAnnealer


class MLPSoftmax(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mlp_params: dict = {},
            conditioning_params: Optional[dict] = None,
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
            
        - conditioning_params: Optional[dict]
            Conditioning-related hyperparameters such as `condition_list` (list of condition names), `use_bias` (whether to use a bias term), `use_residual` (whether to use a residual connection), `residual_weight` (weight of the residual connection), and `init_mode` (initialization mode).
        """
        super().__init__()

        # set parameters of the MLPSoftmax decoder
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_module = ConditionalMLP_Module(
            in_channels=in_channels,
            out_channels=out_channels,
            **mlp_params,
            plain_last=True,
            conditioning_params=conditioning_params,
            apply_to_last=False,
            film_position='post_norm',
        )


    def forward(
            self,
            x: torch.Tensor,
            read_depth: torch.Tensor,
            conditions: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Parameters
        ----------
        - x: torch.Tensor
            The input tensor.
            Dimensions: (batch_size, hidden_channels)
        - read_depth: torch.Tensor
            The read depth tensor.
            Dimensions: (batch_size, 1)
        - conditions: Optional[torch.Tensor]
            The conditions tensor.
            Dimensions: (batch_size, condition_dim)

        Returns:
        -------
        - torch.Tensor:
            Output of the MLP followed by a softmax and a multiplication with the read depth.
        """
        xhat = self.mlp_module(
            x=x,
            conditions=conditions,
        )
        xhat = F.softmax(xhat, dim=-1)

        xhat = xhat * read_depth.unsqueeze(-1)

        return xhat
