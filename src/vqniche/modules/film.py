"""Feature-wise Linear Modulation (FiLM) layer with optional dropout and residual connection.
Supports (N, C) tensors. Optionally supports (N, C, H, W) if broadcast_spatial=True.

Examples:
    # Basic initialization with identity mode (gamma=1, beta=0 at init)
    >>> film_identity = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="identity"
    ... )

    # Delta mode: gamma = 1 + s_gamma * g_raw, beta = s_beta * b_raw
    >>> film_delta = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="delta",
    ...     s_gamma=0.1,  # scale for gamma adjustments
    ...     s_beta=0.05   # scale for beta adjustments
    ... )

    # Tanh bounded mode with custom caps
    >>> film_tanh = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="tanh_bounded",
    ...     tanh_cap_gamma=0.2,  # bounds gamma to [0.8, 1.2]
    ...     tanh_cap_beta=0.05   # bounds beta to [-0.05, 0.05]
    ... )

    # Softplus mode for strictly positive gamma
    >>> film_softplus = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="softplus",
    ...     softplus_eps=1e-3  # minimum value for gamma
    ... )

    # Learn only gamma (beta fixed to 0)
    >>> film_gamma_only = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="gamma_only"
    ... )

    # Learn only beta (gamma fixed to 1)
    >>> film_beta_only = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="beta_only"
    ... )

    # Small random initialization around identity
    >>> film_small_random = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="small_random",
    ...     small_random_std=1e-2
    ... )

    # With additional options
    >>> film_with_options = FiLM(
    ...     in_channels=64,
    ...     condition_dim=32,
    ...     init_mode="delta",
    ...     use_dropout=True,
    ...     dropout_prob=0.1,
    ...     use_residual=True,
    ...     residual_weight=0.5,
    ...     broadcast_spatial=True  # for spatial tensors (N,C,H,W)
    ... )
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer with optional dropout and residual connection.
    Supports (N, C) tensors. Optionally supports (N, C, H, W) if broadcast_spatial=True.

    Init modes supported (choose via `init_mode`):
      - "identity"       : independent of conditions at init; gamma=1, beta=0
      - "dependence"     : strong condition dependence at init; weights=1, bias encodes identity
      - "small_random"   : small random weights around identity bias
      - "delta"          : gamma = 1 + s_gamma * g_raw, beta = s_beta * b_raw   (via forward transform)
      - "tanh_bounded"   : gamma = 1 + s_gamma * tanh(g_raw), beta = s_beta * tanh(b_raw)
      - "softplus"       : gamma = softplus(g_raw) + softplus_eps, beta = s_beta * b_raw
      - "gamma_only"     : learn gamma; beta forced to 0 in forward
      - "beta_only"      : learn beta; gamma forced to 1 in forward
      - "two_head_asym"  : behaves like 'delta' with asymmetric scales (single-head interface here)
      - "mlp_zero_last"  : identity-at-init behavior (same effect as 'identity' for this Linear head)

    Notes:
      * Modes like delta/tanh/softplus modify gamma/beta in `forward` (not just init).
      * `condition_list` is preserved and unused (metadata only).
    """
    def __init__(
            self,
            in_channels: int,
            condition_dim: int,
            condition_list: Optional[List[str]] = None,
            generator_hidden: Optional[List[int]] = None,
            use_bias: bool = True,          # include beta term
            use_dropout: bool = False,      # dropout on generated params
            dropout_prob: float = 0.1,
            use_residual: bool = False,     # add x + residual_weight * FiLM(x)
            residual_weight: float = 1.0,
            broadcast_spatial: bool = False, # if True, accepts (N, C, H, W)
            init_mode: str = "identity",
            s_gamma: float = 0.1,
            s_beta: float = 0.05,
            tanh_cap_gamma: float = 0.2,
            tanh_cap_beta: float = 0.05,
            softplus_eps: float = 1e-3,
            small_random_std: float = 1e-2,
        ):
        super().__init__()

        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.generator_hidden = generator_hidden
        self.condition_list = condition_list or []
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.residual_weight = residual_weight
        self.broadcast_spatial = broadcast_spatial

        # Mode/config
        self.init_mode = init_mode.lower()
        self.s_gamma = s_gamma
        self.s_beta = s_beta
        self.tanh_cap_gamma = tanh_cap_gamma
        self.tanh_cap_beta = tanh_cap_beta
        self.softplus_eps = softplus_eps
        self.small_random_std = small_random_std

        out_dim = in_channels * (2 if use_bias else 1)
        # in __init__, replace single Linear with optional MLP
        if self.generator_hidden and len(self.generator_hidden) > 0:
            layers = []
            dim = condition_dim
            for h in self.generator_hidden:
                layers += [nn.Linear(dim, h), nn.GELU()]
                dim = h
            layers += [nn.Linear(dim, out_dim)]
            self.param_generator = nn.Sequential(*layers)
            last = self.param_generator[-1]
        else:
            self.param_generator = nn.Linear(condition_dim, out_dim)
            last = self.param_generator

        # ---- Initialization per mode (for a single Linear head) ----
        # keep the existing init logic using `last`

        if self.init_mode == "identity" or self.init_mode == "mlp_zero_last":
            # Independent of conditions at init; γ=1, β=0 via bias
            nn.init.zeros_(last.weight)
            if use_bias:
                with torch.no_grad():
                    bias = torch.zeros(out_dim)
                    bias[:in_channels] = 1.0   # gamma = 1
                    # beta = 0
                    last.bias.copy_(bias)
            else:
                nn.init.ones_(last.bias)  # gamma = 1

        elif self.init_mode == "dependence":
            # Strong initial dependence; weights=1, bias encodes identity
            nn.init.ones_(last.weight)
            if use_bias:
                with torch.no_grad():
                    bias = torch.zeros(out_dim)
                    bias[:in_channels] = 1.0
                    last.bias.copy_(bias)
            else:
                nn.init.ones_(last.bias)

        elif self.init_mode == "small_random":
            nn.init.normal_(last.weight, std=self.small_random_std)
            if use_bias:
                with torch.no_grad():
                    bias = torch.zeros(out_dim)
                    bias[:in_channels] = 1.0
                    last.bias.copy_(bias)
            else:
                nn.init.ones_(last.bias)

        elif self.init_mode in {"delta", "tanh_bounded", "softplus", "gamma_only", "beta_only", "two_head_asym"}:
            # Identity-at-init weights/bias; behavior comes from forward transform
            nn.init.zeros_(last.weight)
            if use_bias:
                with torch.no_grad():
                    bias = torch.zeros(out_dim)
                    bias[:in_channels] = 1.0
                    last.bias.copy_(bias)
            else:
                nn.init.ones_(last.bias)

        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")

        self.dropout = nn.Dropout(p=dropout_prob) if use_dropout else None


    def _split_params(
            self,
            film_params: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_bias:
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
        else:
            gamma, beta = film_params, None
        return gamma, beta


    def _apply_mode_transform(
            self,
            gamma: torch.Tensor,
            beta: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply per-mode transformations to raw generator outputs.
        """
        mode = self.init_mode

        if mode in {"identity", "dependence", "small_random", "mlp_zero_last"}:
            # Use raw outputs (already identity-biased at init)
            return gamma, beta

        if mode in {"delta", "two_head_asym"}:
            gamma = 1.0 + self.s_gamma * gamma
            if self.use_bias and beta is not None:
                beta = self.s_beta * beta
            return gamma, beta

        if mode == "tanh_bounded":
            gamma = 1.0 + self.tanh_cap_gamma * torch.tanh(gamma)
            if self.use_bias and beta is not None:
                beta = self.tanh_cap_beta * torch.tanh(beta)
            return gamma, beta

        if mode == "softplus":
            gamma = F.softplus(gamma) + self.softplus_eps
            if self.use_bias and beta is not None:
                beta = self.s_beta * beta
            return gamma, beta

        if mode == "gamma_only":
            # Learn γ; force β=0
            if self.use_bias and beta is not None:
                beta = torch.zeros_like(beta)
            return gamma, beta

        if mode == "beta_only":
            # Learn β; force γ=1
            gamma = torch.ones_like(gamma)
            return gamma, beta

        # Fallback (shouldn't reach here)
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

        # Mode-specific transform (delta/tanh/softplus/etc.)
        gamma, beta = self._apply_mode_transform(gamma, beta)

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

        # FIX: residual mode previously returned `x + residual_weight * x_conditioned`,
        # which with identity init (gamma=1, beta=0) yields
        #   output = x + residual_weight * x = (1 + residual_weight) * x
        # i.e. the layer is NOT the identity at init even though `init_mode='identity'`
        # was supposed to make it so. We now return the *delta* relative to the
        # identity, so the residual mixes only the modulation signal:
        #   output = x + residual_weight * (x_conditioned - x)
        # With identity init this correctly yields `output = x` regardless of
        # `residual_weight`.
        if self.use_residual:
            return x + self.residual_weight * (x_conditioned - x)
        return x_conditioned
