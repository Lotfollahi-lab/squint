"""
Consider a GraphSAGE module with 2 layers. The forward method of the module returns h_v^1 as the output of the first layer and h_v^2. From Step 5 (Algorithm 1) of the paper, the output of the first layer is computed as follows:
`h_v^1 = ReLU(W_0 * CONCAT(x_v, sum_{u \in N(v)} x_u))`

The output of the second layer is computed as follows:
`h_v^2 = W_1 * CONCAT(h_v^1, sum_{u \in N(v)} h_u^1)`

Setting number of features = in_channels, internal dimension = hidden_channels, and number of classes = out_channels, Pytorch Geometric implements a 2-layer GraphSAGE module by using 2 SAGEConv modules.

If we set the layers as follows (and Jumping Knowledge is not used):
Layer 1 = SAGEConv(in_channels, hidden_channels)
Layer 2 = SAGEConv(hidden_channels, out_channels)
Then, internally, the forward of GraphSAGE applies the following operations:
Layer 1:
    - Forward of SAGEConv Layer 1:
        - Message passing step: in_channels -> in_channels
        - Linear transformation (P_0): in_channels -> hidden_channels
    - RELU activation
    - Dropout
Layer 2:
    - Forward of SAGEConv Layer 2:
        - Message passing step: hidden_channels -> hidden_channels
        - Linear transformation (P_1): hidden_channels -> out_channels
    (No activation function is applied in the last layer)
This means we cannot directly access the internal node embeddings of the GraphSAGE module. We also cannot set the initializations of the Linear layers of the SAGEConv module.

So we implement a custom GNN module that inherits from Pytorch Geometric's SAGEConv module.

More specifically, we do the following:
GraphSAGE encoder:
Layer 1 = SAGEConv(in_channels, hidden_channels)
Layer 2 = SAGEConv(hidden_channels, hidden_channels)
Internally, this applies the following operations:
Layer 1:
    - Forward of SAGEConv Layer 1:
        - Message passing step: in_channels -> in_channels
        - Linear transformation (P_0): in_channels -> hidden_channels
    - RELU activation
    - Dropout
Layer 2:
    - Forward of SAGEConv Layer 2:
        - Message passing step: hidden_channels -> hidden_channels
        - Linear transformation (P_3): hidden_channels -> hidden_channels
    (No activation function is applied in the last layer)

Because P_3 and P_4 are both learnable parameters of the model, they will be updated during training. In matrix form, P_3 * P_4 will have the same dimensions as P_1 which is the linear transformation in the last layer of the Pytorch Geometric's GraphSAGE model. Mathematically, this is the same as applying the linear transformation P_1 in the last layer of the Pytorch Geometric's GraphSAGE model. The only downside is that we need to maintain an extra matrix of trainable parameters which impacts the memory usage of the model.

This file implements a drop-in replacement for Pytorch Geometric's SAGE, GATv2, or GIN modules either as an encoder for Vanilla GNN Models or as a base GNN module for more complex VQ-GNN models. The predictors and decoders are implemented by the respective Model classes.

Further, this module adds two customization functionalities:
1. A series of Linear layers to transform the input features before the GNN layers.
2. Choice of initialization method for the GNN's Linear layers between kaiming_uniform, glorot, or uniform.
"""

from typing import Literal, Union, Callable, List

import torch
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GraphSAGE as BaseSAGEConv_Module
from torch_geometric.nn import GAT as BaseGATv2_Module
from torch_geometric.nn import GIN as BaseGIN_Module
from torch_geometric.nn.aggr import MultiAggregation


def init_gnn_module(
        in_channels: int = None,
        gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv',
        gnn_params: dict = {},
    ) -> torch.nn.Module:
    """
    Initialize the GNN module.

    Parameters
    ----------
    - in_channels: int
        The number of input channels.
    - gnn_name: str
        The name of the GNN module.
    - gnn_params: dict
        The parameters for the GNN module.

    Returns
    -------
    - GNN_Module: torch.nn.Module
        The GNN module.
    """
    GNN_Module = create_dynamic_gnn_module_class(gnn_name=gnn_name)

    return GNN_Module(
        in_channels=in_channels,
        **gnn_params,
    )


def create_dynamic_gnn_module_class(
        gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv'] = 'SAGEConv'
    ) -> torch.nn.Module:
    """
    Create a GNN Module class dynamically based on the name.

    Parameters
    ----------
    gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv']
        The name of the GNN module.

    Returns
    -------
    - torch.nn.Module
        The GNN module.
    """
    # set the base GNN module class from Pytorch Geometric
    if gnn_name == 'SAGEConv':
        BaseGNN_Module = BaseSAGEConv_Module
    elif gnn_name == 'GATv2Conv':
        BaseGNN_Module = BaseGATv2_Module
    elif gnn_name == 'GINConv':
        BaseGNN_Module = BaseGIN_Module

    # define the VanillaGNN_Module class that inherits from the base GNN module
    # this adds two customizations to the BaseGNN_Module:
    # 1. a series of Linear layers to transform the input features before the GNN layers
    # 2. an initialization method for the GNN's Linear layers
    class VanillaGNN_Module(BaseGNN_Module):

        def __init__(
            self,
            in_channels: int = None,
            hidden_channels: List[int] | int = 500,
            num_layers: int = 2,
            act_first: bool = True,
            activation: Union[str, Callable, None] = "relu",
            norm: Union[str, Callable, None] = None,
            dropout: float = 0.5,
            init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None] = 'kaiming_uniform',
        ):
            """
            Initialize the GNN module.

            Parameters
            ----------
            - in_channels: int
                The number of input channels.
            - hidden_channels: List[int] | int
                The number of hidden channels.
            - num_layers: int
                The number of GNN layers.
            - act_first: bool
                Whether to apply the activation function before the GNN layer.
            - activation: Union[str, Callable, None]
                The activation function.
            - norm: Union[str, Callable, None]
                The normalization function.
            - dropout: float
                The dropout rate.
            - init_method: Literal['kaiming_uniform', 'glorot', 'uniform', None]
                The initialization method.
            """
            self.num_layers = num_layers

            kwargs = {
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'num_layers': num_layers,
                'act_first': act_first,
                'act': activation,
                'dropout': dropout,
                'norm': norm
            }
            if isinstance(self, BaseGATv2_Module):
                kwargs['v2'] = True

            super().__init__(**kwargs)

            if init_method is None:
                init_method = 'kaiming_uniform'
            self.init_method = init_method
            self.initialize_linear_layers()

            self.gnn_layer_name = gnn_name
            self.dim = hidden_channels


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

    return VanillaGNN_Module
