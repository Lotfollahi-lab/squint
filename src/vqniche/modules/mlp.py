"""
MLP module.
Source: Amir
"""
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        layer_norm: bool = False,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ln = nn.LayerNorm(hidden_features) if layer_norm else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x