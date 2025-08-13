from typing import Literal, Optional, List

import torch
import pytorch_lightning as pl

from vqniche.modules import MLP as MLP_Module
from vqniche.modules import init_gnn_module
from vqniche.modules import FiLM
from vqniche.modules import get_vq_class, get_valid_params


class VQNiche_Encoder(pl.LightningModule):

    def __init__(
            self,
            in_channels: int = None,
            mlp_params: Optional[dict] = None,
            gnn_name: Optional[Literal['SAGEConv', 'GATv2Conv', 'GINConv']] = None,
            gnn_params: dict = {},
            conditioning_params: dict = {},
            vq_params: dict = {},
        ):
        """
        Initialize the VQGraph_Encoder.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - mlp_params: Optional[dict]
            Keyword arguments for the MLP module.
            Default: None. If None, the MLP module will not be used.
        - gnn_name: Literal['SAGEConv', 'GATv2Conv', 'GINConv']
            The name of the GNN module.
        - gnn_params: dict
            Keyword arguments for the GNN module.
        - conditioning_params: dict
            Keyword arguments for the conditioning module.
        - vq_params: dict
            Keyword arguments for the VQ module.
        """
        super().__init__()

        # if mlp_params is not provided, the MLP module will not be used
        if mlp_params is None:
            self.mlp_module = None

            # the GNN module will use the input channels as the input channels
            gnn_in_channels = in_channels
            
            self.mlp_layers = 0
        else:
            self.mlp_module = MLP_Module(
                in_channels=in_channels,
                **mlp_params,
            )

            # the GNN module will use the output channels of the MLP module as the input channels
            gnn_in_channels = self.mlp_module.out_channels

            self.mlp_layers = self.mlp_module.num_layers

        # initialize the GNN module
        if gnn_params['num_layers'] == 0:
            self.gnn_module = None
            vq_params['dim'] = gnn_in_channels
            self.gnn_layers = 0
        else:
            self.gnn_module = init_gnn_module(
                in_channels=gnn_in_channels,
                gnn_name=gnn_name,
                gnn_params=gnn_params,
            )
            vq_params['dim'] = self.gnn_module.dim
            self.gnn_layers = self.gnn_module.num_layers

        assert self.mlp_layers > 0 or self.gnn_layers > 0, "Both MLP and GNN modules have 0 layers. Please set at least one of the num_layers to a positive integer."

        # initialize the conditioning module
        if 'condition_list' in conditioning_params:
            self.conditioning_module = FiLM(
                in_channels=vq_params['dim'],
                **conditioning_params,
            )
        else:
            self.conditioning_module = None

        # initialize the vq module
        self.vq = self._init_vq_module(
                    vq_params_dict=vq_params
                    )

        self.dim = vq_params['dim']


    def _init_vq_module(
            self,
            vq_params_dict: dict,
        ):
        VQ_Module = get_vq_class(vq_params_dict['vq_name'])
        valid_vq_params = get_valid_params(
                            VQ_Module,
                            vq_params_dict,
                        )
        return VQ_Module(
                **valid_vq_params,
            )


    def forward(
            self,
            batch_x: torch.Tensor,
            batch_edge_index: torch.Tensor,
            batch_encoder_conditions: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Forward pass of the VQGraph_Encoder model.

        Parameters:
        ----------
        - batch_x: torch.Tensor
            The input features of the batch of nodes.
        - batch_edge_index: torch.Tensor
            The edge index tensor of the batch of nodes.
        - batch_encoder_conditions: Optional[torch.Tensor]
            The conditioning features for the encoder of the batch of nodes.

        Returns
        -------
        - h_latent: torch.Tensor
            Forward (output) of just the MLP or just the GNN module or MLP followed by the GNN module.
        - h_quantized: torch.Tensor
            Forward (output) of the VQ module.
        - indices: torch.Tensor
            The indices of the node embeddings mapped to codebook embeddings.
        """
        # forward pass of the MLP module
        if self.mlp_module is not None:
            h_mlp = self.mlp_module(batch_x)
        else:
            h_mlp = batch_x

        # forward pass of the GNN module
        if self.gnn_module is not None:
            h_latent = self.gnn_module(
                h_mlp,
                batch_edge_index
            )
        else:
            h_latent = h_mlp

        # forward pass of the conditioning module
        if self.conditioning_module is not None:
            h_latent = self.conditioning_module(
                x=h_latent,
                conditions=batch_encoder_conditions,
            )

        # VQ-encode the node embeddings
        h_quantized, \
        indices, \
        _, \
            = self.vq(h_latent)

        return h_latent, \
            h_quantized, \
            indices