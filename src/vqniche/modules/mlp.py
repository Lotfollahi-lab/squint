from typing import List, Union, Callable, Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
import pytorch_lightning as pl


class MLP_Module(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            num_layers: int,
            hidden_channels: Union[int, List[int]],
            activation: str = 'relu',
            dropout: float = 0.0,
            norm: Optional[Literal['BatchNorm1d', 'LayerNorm']] = None,
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
        ):
        """
        Multi-Layer Perceptron (MLP) module.

        Parameters
        ----------
        - in_channels : int
            Number of input features
        - num_layers : int
            Number of linear layers (excluding input and output layers)
        - hidden_channels : Union[int, List[int]]
            Number of hidden features. If int, all hidden layers will have this many features.
            If list, must have length num_layers and specifies the number of features for each hidden layer.
        - activation : str, optional
            Activation function to use. One of: 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu'
        - dropout : float, optional
            Dropout probability. Default: 0.0
        - norm : Optional[Literal['BatchNorm1d', 'LayerNorm']], optional
            Normalization to use. One of: 'BatchNorm1d', 'LayerNorm', None. Default: None
        - init_method : Literal['kaiming_uniform', 'glorot', 'uniform', None], optional
            Initialization method to use. Default: 'kaiming_uniform'
        """
        if num_layers == 0:
            return None

        super().__init__()

        # Validate parameters
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers)
        elif isinstance(hidden_channels, list):
            if len(hidden_channels) != num_layers:
                raise ValueError(f"Expected hidden_channels list of length {num_layers}, got {len(hidden_channels)}")

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dim = self.hidden_channels[-1]
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = norm
        self.init_method = init_method

        # Build layers
        self.layers, self.norms = self._build_layers_and_norms()


    def _build_layers_and_norms(self) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Build layers and norms.

        Returns
        -------
        - layers : nn.ModuleList
            List of linear layers.
        - norms : nn.ModuleList
            List of normalization layers.
        """
        layers = nn.ModuleList()
        norms = nn.ModuleList()

        # Build layers and norms
        for i in range(self.num_layers):
            if i == 0:
                in_dim = self.in_channels
                out_dim = self.hidden_channels[0]
            else:
                in_dim = self.hidden_channels[i-1]
                out_dim = self.hidden_channels[i]

            layers.append(
                Linear(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    weight_initializer=self.init_method
                )
            )

            if i < self.num_layers-1:  # Don't add norm after final layer
                if self.norm is not None:
                    if self.norm == 'BatchNorm1d':
                        norms.append(
                            nn.BatchNorm1d(
                                num_features=out_dim
                            )
                        )
                    elif self.norm == 'LayerNorm':
                        norms.append(
                            nn.LayerNorm(
                                normalized_shape=out_dim
                            )
                        )
                else:
                    norms.append(nn.Identity())

        return layers, norms


    def _get_activation(
            self,
            activation: str
        ) -> Callable:
        """
        Get activation function from string name.

        Parameters
        ----------
        - activation : str
            Activation function to use.

        Returns
        -------
        - activation_fn : Callable
            Activation function.
        """
        if activation == 'relu':
            return F.relu
        elif activation == 'leaky_relu':
            return F.leaky_relu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Unknown activation function: {activation}")


    def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        - x : torch.Tensor
            Input tensor of shape [batch_size, in_channels]

        Returns
        -------
        - output : torch.Tensor
            Output tensor of shape [batch_size, hidden_channels[-1]]
        """
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # No norm, activation, dropout on final layer
        x = self.layers[-1](x)

        return x
