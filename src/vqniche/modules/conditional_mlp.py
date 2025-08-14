from typing import Optional, List

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP as MLP_Module
from .film import FiLM


class ConditionalMLP(MLP_Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        hidden_channels: List[int] = [],
        dropout: float = 0.0,
        act: str = 'relu',
        norm: Optional[str] = None,
        plain_last: bool = True,
        conditioning_params: Optional[dict] = None,  # e.g., {'condition_dim': D, 'init_mode': 'delta', ...}
        film_position: str = 'post_norm',            # 'pre_norm' | 'post_norm'
        apply_to_last: bool = False,
    ):
        # build same as current MLP
        if in_channels is not None and out_channels is None:
            channel_list = [in_channels] + hidden_channels
        elif in_channels is None and out_channels is not None:
            channel_list = hidden_channels + [out_channels]
        elif in_channels is not None and out_channels is not None:
            channel_list = [in_channels] + hidden_channels + [out_channels]
        else:
            channel_list = hidden_channels
        assert len(channel_list) > 1

        super().__init__(
            channel_list=channel_list,
            dropout=dropout,
            act=act,
            act_first=False,
            norm=norm,
            plain_last=plain_last,
        )

        self.film_position = film_position
        self.apply_to_last = apply_to_last
        self.films = None
        if conditioning_params is not None:
            self.films = []
            n_apply = len(self.lins) - (0 if self.apply_to_last else 1)
            for i, lin in enumerate(self.lins):
                if i < n_apply:
                    self.films.append(
                        FiLM(in_channels=lin.out_channels, **conditioning_params)
                    )
            self.films = torch.nn.ModuleList(self.films)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: Optional[bool] = None,
        conditions: Optional[torch.Tensor] = None,  # optional
    ) -> torch.Tensor:
        emb: Optional[torch.Tensor] = None
        film_on = self.films is not None and conditions is not None

        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if film_on and self.film_position == 'post_norm' and i < len(self.films):
                x = self.films[i](x, conditions)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            if film_on and self.film_position == 'pre_norm' and i < len(self.films):
                x = self.films[i](x, conditions)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            if film_on and self.apply_to_last and len(self.films) > len(self.norms):
                x = self.films[-1](x, conditions)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x