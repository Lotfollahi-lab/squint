from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from vqniche.modules.mlp import MLP as MLP_Module
from vqniche.modules.film import FiLM
from .temperature_annealer import TemperatureAnnealer


class MLPSoftmax(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mlp_params: dict = {},
            conditioning_params: dict = {},
            temperature_annealer_params: Optional[dict] = None,
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
            
        - temperature_annealer_params: dict
            Temperature annealer-related hyperparameters such as `start_temp` (initial temperature), `end_temp` (final temperature), `total_steps` (number of steps over which to anneal), and `mode` (mode of temperature annealing).
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
        
        if temperature_annealer_params is not None:
            self.temperature_annealer = TemperatureAnnealer(
                **temperature_annealer_params,
            )
        
        if 'condition_list' in conditioning_params:
            self.conditioning_module = FiLM(
                in_channels=in_channels,
                **conditioning_params,
            )
        else:
            self.conditioning_module = None


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
        if self.conditioning_module is not None:
            x = self.conditioning_module(
                    x=x,
                    conditions=conditions,
                )

        xhat = self.mlp_module(x)
        
        if hasattr(self, 'temperature_annealer'):
            temp = self.temperature_annealer.step()
        else:
            temp = 1.0
        xhat = F.softmax(xhat / temp, dim=-1)

        xhat = xhat * read_depth.unsqueeze(-1)

        return xhat
