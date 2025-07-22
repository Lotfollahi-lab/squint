from typing import Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl


class FiLM(pl.LightningModule):
    """
    Feature-wise Linear Modulation (FiLM) module.
    
    This module applies an affine transformation to input features, where the 
    transformation parameters (gamma for scaling, beta for shifting) are computed 
    from conditioning inputs.
    """
    def __init__(
            self,
            in_channels: int,
            condition_dim: int,
            use_bias: bool = True,
        ):
        """
        Initialize the FiLM module.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels/features to be modulated
        condition_dim : int
            Dimension of the conditioning input
        use_bias : bool, optional
            Whether to include a bias (beta) term in the FiLM transformation
            Default: True
        """
        super().__init__()
        
        self.name = 'FiLM'
        
        # Layer to generate FiLM parameters from conditioning input
        film_param_dim = in_channels * (2 if use_bias else 1)
        self.param_generator = nn.Linear(
            condition_dim,
            film_param_dim
        )
        
        self.in_channels = in_channels
        self.use_bias = use_bias
        
        # Initialize parameters to perform identity transformation
        nn.init.ones_(self.param_generator.weight)
        nn.init.zeros_(self.param_generator.bias)


    def forward(
            self,
            x: torch.Tensor,
            conditions: torch.Tensor,
        ) -> torch.Tensor:
        """
        Apply FiLM transformation to the input features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features to be modulated
            Shape: (batch_size, in_channels)
        conditions : torch.Tensor
            Conditioning inputs used to generate FiLM parameters
            Shape: (batch_size, condition_dim)
            
        Returns
        -------
        torch.Tensor
            FiLM-modulated features
            Shape: (batch_size, in_channels)
        """
        # Generate FiLM parameters
        film_params = self.param_generator(conditions)
        
        # Split into gamma and beta
        if self.use_bias:
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
        else:
            gamma = film_params
            beta = None
        
        # Apply FiLM transformation
        x_conditioned = gamma * x
        if self.use_bias:
            x_conditioned = x_conditioned + beta
            
        return x_conditioned
