from typing import Literal, List

import torch
import torch.nn as nn
import pytorch_lightning as pl


class FiLM(pl.LightningModule):
    """
    Feature-wise Linear Modulation (FiLM) module with optional dropout and residual connection.
    """
    def __init__(
            self,
            in_channels: int,
            condition_list: List[str] = [],
            use_bias: bool = True,
            use_dropout: bool = False,
            dropout_prob: float = 0.1,
            use_residual: bool = False,
            residual_weight: float = 1.0,
        ):
        """
        Initialize the FiLM module.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels/features to be modulated
        condition_list : List[str], optional
            List of condition names to be used for conditioning
        use_bias : bool, optional
            Whether to include a bias (beta) term in the FiLM transformation
        use_dropout : bool, optional
            Whether to apply dropout to gamma and beta
        dropout_prob : float, optional
            Dropout probability (only used if use_dropout is True)
        use_residual : bool, optional
            Whether to add residual connection from input x
        residual_weight : float, optional
            Weight of the FiLM-transformed term in the residual sum
        """
        super().__init__()
        
        self.name = 'FiLM'
        self.in_channels = in_channels
        self.condition_list = condition_list
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.residual_weight = residual_weight
        
        self.film_param_dim = in_channels * (2 if use_bias else 1)
        self.param_generator = None  # Will be initialized in setup()

        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_prob)


    def setup(
            self,
            condition_dim: int,
        ) -> None:
        """
        Initialize the parameter generator once condition_dim is known.
        
        Parameters
        ----------
        condition_dim : int
            Dimension of the conditioning input
        """
        self.param_generator = nn.Linear(
            condition_dim,
            self.film_param_dim,
        ).to(self.device)

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
            Input features to be modulated (N, in_channels)
        conditions : torch.Tensor
            Conditioning inputs (N, condition_dim)
            
        Returns
        -------
        torch.Tensor
            FiLM-modulated features (N, in_channels)
        """
        if self.param_generator is None:
            self.setup(conditions.shape[-1])
        
        film_params = self.param_generator(conditions)

        if self.use_bias:
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
        else:
            gamma = film_params
            beta = None

        if self.use_dropout:
            gamma = self.dropout(gamma)
            if beta is not None:
                beta = self.dropout(beta)

        x_conditioned = gamma * x
        if self.use_bias:
            x_conditioned = x_conditioned + beta

        if self.use_residual:
            # Combine original and conditioned features
            return x + self.residual_weight * x_conditioned
        else:
            return x_conditioned