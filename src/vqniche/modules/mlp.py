from torch_geometric.nn import MLP as MLP_Module


class MLP(MLP_Module):
    def __init__(
            self,
            in_channels: int = None,
            out_channels: int = None,
            mlp_params: dict = {},
        ):
        """
        Initialize the MLP module.

        Parameters
        ----------
        - in_channels: int
            The number of input channels.
        - out_channels: int
            The number of output channels.
        - mlp_params: dict
            The parameters for the MLP module.

        Notes
        -----
        - If `in_channels` is not provided, the MLP module will assume the input channel is `mlp_params['hidden_channels'][0]`.
        - If `out_channels` is not provided, the MLP module will assume the output channel is `mlp_params['hidden_channels'][-1]`.
        """
        if in_channels is None and out_channels is None and mlp_params['hidden_channels'] is None:
            return

        if mlp_params['hidden_channels'] is None and not (in_channels is not None and out_channels is not None):
            raise ValueError("If `mlp_params['hidden_channels']` is not provided, `in_channels` and `out_channels` must be provided.")

        if mlp_params['hidden_channels'] is not None:
            channel_list = mlp_params['hidden_channels']
        else:
            channel_list = []

        if in_channels is not None:
            channel_list = [in_channels] + channel_list
        if out_channels is not None:
            channel_list = channel_list + [out_channels]

        super().__init__(
            channel_list=channel_list,
            dropout=mlp_params['dropout'],
            act=mlp_params['act'],
            norm=mlp_params['norm'],
        )