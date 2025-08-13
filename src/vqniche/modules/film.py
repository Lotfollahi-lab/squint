from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer with optional dropout and residual connection.
    Supports (N, C) tensors. Optionally supports (N, C, H, W) if broadcast_spatial=True.
    """
    def __init__(
            self,
            in_channels: int,
            condition_dim: int,
            condition_list: Optional[List[str]] = None,
            weight_init: float = 1.0,
            use_bias: bool = True,          # include beta term
            use_dropout: bool = False,      # dropout on generated params
            dropout_prob: float = 0.1,
            use_residual: bool = False,     # add x + residual_weight * FiLM(x)
            residual_weight: float = 1.0,
            broadcast_spatial: bool = False # if True, accepts (N, C, H, W)
        ):
        super().__init__()

        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.condition_list = condition_list or []
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.residual_weight = residual_weight
        self.broadcast_spatial = broadcast_spatial

        out_dim = in_channels * (2 if use_bias else 1)
        self.param_generator = nn.Linear(condition_dim, out_dim)

        # Weight initialization:
        # Make output dependent on conditions at init (weights=1)
        # this tells the model to start with a strong dependence on conditions
        if weight_init == 1.0:
            nn.init.ones_(self.param_generator.weight)

        # Make output independent of conditions at init (weights=0)
        # this tells the model to start with no dependence on conditions
        elif weight_init == 0.0:
            nn.init.zeros_(self.param_generator.weight)

        else:
            raise ValueError(f"Invalid weight_init: {weight_init}")

        # set bias so that gamma=1, beta=0 (if use_bias=True).
        if use_bias:
            with torch.no_grad():
                bias = torch.zeros(out_dim)
                bias[:in_channels] = 1.0   # gamma = 1
                bias[in_channels:] = 0.0   # beta  = 0
                self.param_generator.bias.copy_(bias)
        else:
            nn.init.ones_(self.param_generator.bias)  # gamma = 1

        self.dropout = nn.Dropout(p=dropout_prob) if use_dropout else None


    def _split_params(
            self,
            film_params
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_bias:
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
        else:
            gamma, beta = film_params, None
        return gamma, beta


    def forward(
            self,
            x: torch.Tensor,
            conditions: torch.Tensor,
        ) -> torch.Tensor:
        """
        x: (N, C) or (N, C, H, W) if broadcast_spatial=True
        conditions: (N, condition_dim)
        """
        film_params = self.param_generator(conditions)   # (N, C) or (N, 2C)
        gamma, beta = self._split_params(film_params)

        if self.use_dropout and self.training:
            gamma = self.dropout(gamma)
            if beta is not None:
                beta = self.dropout(beta)

        if self.broadcast_spatial:
            # reshape to (N, C, 1, 1) to broadcast over H, W
            shape = [x.size(0), self.in_channels] + [1] * (x.dim() - 2)
            gamma = gamma.view(*shape)
            if beta is not None:
                beta = beta.view(*shape)

        x_conditioned = gamma * x
        if beta is not None:
            x_conditioned = x_conditioned + beta

        return x + self.residual_weight * x_conditioned if self.use_residual else x_conditioned