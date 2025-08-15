from typing import Optional, List

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP as MLP_Module


class MLP(MLP_Module):
    def __init__(
            self,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            hidden_channels: List[int] = [],
            dropout: float = 0.0,
            act: str = 'relu',
            norm: Optional[str] = None,
            plain_last: bool = True,
        ):
        """
        Initialize the MLP module.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - hidden_channels: List[int]
            The number of hidden channels representing the number of dimensions of the hidden features in the intermediate layers of the MLP.
        - dropout: float
            The dropout rate.
        - act: str
            The activation function.
        - norm: Optional[str]
            The normalization function.
        - plain_last: bool
            Whether to apply non-linearity, batch normalization and dropout to the last layer.

        Notes
        -----
        - If `in_channels` is not provided, the MLP module will assume the input channel is `mlp_params['hidden_channels'][0]`.
        - If `out_channels` is not provided, the MLP module will assume the output channel is `mlp_params['hidden_channels'][-1]`.
        """
        # if in_channels is provided, but out_channels is not, the MLP module will assume the output channel is `mlp_params['hidden_channels'][-1]`.
        if in_channels is not None and out_channels is None:
            channel_list = [in_channels] + hidden_channels
            
        # if out_channels is provided, but in_channels is not, the MLP module will assume the input channel is `mlp_params['hidden_channels'][0]`.
        elif in_channels is None and out_channels is not None:
            channel_list = hidden_channels + [out_channels]
        
        # if both in_channels and out_channels are provided, the MLP module will use them as the input and output channels.
        elif in_channels is not None and out_channels is not None:
            channel_list = [in_channels] + hidden_channels + [out_channels]

        # if both in_channels and out_channels are not provided, use hidden_channels as channel_list
        else:
            channel_list = hidden_channels

        assert len(channel_list) > 1, f"Channel list has length {len(channel_list)} which is less than 2. Please provide at least an input channel and an output channel."
        
        # initialize the MLP module using channel_list so that hidden layers of different dimensions can be used
        # e.g. 200 -> [400, 600] -> 1000 = 3 layers
        super().__init__(
            channel_list=channel_list,
            dropout=dropout,
            act=act,
            act_first=False,
            norm=norm,
            plain_last=plain_last,
        )
        

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of the
                final output layer. (default: :obj:`False`)
        """
        # `return_emb` is annotated here as `NoneType` to be compatible with
        # TorchScript, which does not support different return types based on
        # the value of an input argument.
        emb: Optional[torch.Tensor] = None

        # If `plain_last=True`, then `len(norms) = len(lins) -1, thus skipping
        # the execution of the last layer inside the for-loop.
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x