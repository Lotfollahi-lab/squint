"""
BaseSAGEConv_Module is torch_geometric.nn.GraphSAGE and it inherits from torch_geometric.nn.BasicGNN. It is intended to be used as a Model by itself. However, it takes away some flexibility in the initialization of the linear layers. We import it as BaseSAGEConv_Module and inherit from it so that we can get the benefits of the BaseSAGEConv_Module and also have the flexibility to specify the initialization method of the linear layers.

This is now usable as either:
- GraphSAGE_Encoder in our implementation of the GraphSAGE Model in the models.graphsage.py file.
- GraphSAGE_Module in our implementation of the VQGraph Model in the encoders.vqgraph_encoder.py file.

In the init method of BasicGNN, init_conv is set to torch_geometric.nn.conv.SAGEConv (in the case of torch_geometric.nn.GraphSAGE) which is the actual GraphSAGE convolution layer. SAGEConv inherits from torch_geometric.nn.conv.MessagePassing which is a base class for message passing layers. Within it's init method, SAGEConv builds lin_l and lin_r as torch.nn.Linear layers with the kaiming_uniform initialization method by default. We re-initialize with glorot or uniform initialization method if specified and do nothing if init_method is None or kaiming_uniform.
"""

from typing import Literal, Union, Callable, List

import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GraphSAGE as BaseSAGEConv_Module
from torch_geometric.nn.aggr import MultiAggregation


class SAGEConv_Module(BaseSAGEConv_Module):
    def __init__(
            self,
            num_linear_layers: int = 2,
            in_channels: int = None,
            hidden_channels: List[int] | int = 500,
            num_gnn_layers: int = 2,
            act_first: bool = True,
            activation: Union[str, Callable, None] = "relu",
            norm: Union[str, Callable, None] = None,
            dropout: float = 0.5,
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
        ):
        """
        Initializes the SAGE_Conv_Module.

        Parameters
        ----------
        - num_linear_layers: int
            The number of linear layers to use before the GraphSAGE encoder layers.
        - in_channels: int
            The number of input features.
        - hidden_channels: List[int] | int
            The number of hidden features.
        - num_gnn_layers: int
            The number of GraphSAGE encoder layers.
        - act_first: bool
            Whether to apply the activation function before normalization.
        - activation: str or callable or None
            The activation function to use.
        - norm: str or callable or None
            The normalization function to use.
        - dropout: float
            The dropout probability.
        - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
            The initialization method to use for the linear transformations in the SAGEConv layers.
            If None, the initialization method is 'kaiming_uniform'.
        """
        self.num_linear_layers = num_linear_layers
        if num_linear_layers > 0:
            if isinstance(hidden_channels, int):
                hidden_channels = [in_channels] + [hidden_channels] * num_linear_layers
            elif isinstance(hidden_channels, list):
                assert len(hidden_channels) == num_linear_layers, f"The number of hidden channels must be equal to the number of linear layers. Got {len(hidden_channels)} hidden channels for {num_linear_layers} linear layers."
                hidden_channels = [in_channels] + hidden_channels

            print(f"{num_linear_layers=} | {hidden_channels=}")

            # if Linear layers are used, SAGEConv is applied from the last hidden channel to the last hidden channel
            sageconv_in_channels = hidden_channels[-1]
            sageconv_hidden_channels = hidden_channels[-1]
        else:
            # if no Linear layers are used, SAGEConv is applied from the input channels to hidden channels
            sageconv_in_channels = in_channels
            sageconv_hidden_channels = hidden_channels

        super().__init__(
            in_channels=sageconv_in_channels,
            hidden_channels=sageconv_hidden_channels,
            num_layers=num_gnn_layers,
            act_first=act_first,
            act=activation,
            dropout=dropout,
            norm=norm,
        )

        if init_method is None:
            init_method = 'kaiming_uniform'
        self.init_method = init_method
        self.initialize_linear_layers()

        if num_linear_layers > 0:
            # Create sequential model of linear layers
            layers = []
            for i in range(num_linear_layers):
                layers.append(Linear(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i+1],
                    bias=True,
                    weight_initializer=init_method,
                    bias_initializer=None,
                ))

            self.input_transform = nn.Sequential(*layers)


    def initialize_linear_layers(self):
        """
        Initializes the linear layers of the SAGEConv module with the desired initialization method.
        """
        print(f"Initializing linear layers with {self.init_method} initialization method.")
        if self.init_method == 'kaiming_uniform':
            pass
        elif self.init_method in ['glorot', 'uniform']:
            for conv in self.convs:
                in_channels = conv.in_channels
                if isinstance(in_channels, int):
                    in_channels = (in_channels, in_channels)
                if isinstance(conv.aggr_module, MultiAggregation):
                    lin_l_in_channels = conv.aggr_module.get_out_channels(in_channels[0])
                else:
                    lin_l_in_channels = in_channels[0]

                conv.lin_l = Linear(
                    in_channels=lin_l_in_channels,
                    out_channels=conv.out_channels,
                    bias=True,
                    weight_initializer=self.init_method,
                    bias_initializer=None,
                )
                if conv.lin_r is not None:
                    conv.lin_r = Linear(
                        in_channels=in_channels[1],
                        out_channels=conv.out_channels,
                        bias=False,
                        weight_initializer=self.init_method,
                        bias_initializer=None,
                    )
        else:
            raise ValueError(f"Invalid initialization method: {self.init_method}")


    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the SAGEConv module.

        Parameters
        ----------
        - x: torch.Tensor
            The input features.
        - edge_index: torch.Tensor
            The edge index.

        Returns
        -------
        - torch.Tensor
            The output features.
        """
        if self.num_linear_layers > 0:
            x = self.input_transform(x)

        return super().forward(
            x,
            edge_index
        )