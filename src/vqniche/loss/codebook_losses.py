"""
Basic codebook loss functions for vector quantization.
"""

import torch
import torch.nn.functional as F


def mse_commit_loss(
        quantizer_input: torch.Tensor,
        quantizer_output: torch.Tensor,
        wt_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Compute the commit loss.

    Parameters
    ----------
    quantizer_input: torch.Tensor
        Input to the VQ module.
        Dimensions: (batch_size, hidden_channels)
    quantizer_output: torch.Tensor
        Output from the VQ module.
        Dimensions: (batch_size, hidden_channels)
    wt_commit: float
        The scaling factor for the commitment loss.

    Returns
    -------
    commit_loss: torch.Tensor
        The computed commit loss.

    Notes:
    -----
    - The quantizer_input is latent node embeddings from the GNN module.
    - The quantizer_output is quantized node embeddings, i.e. the nearest codebook embedding for each node.
    - This loss function freezes the codebook embeddings and updates the node embeddings.
    - Reference --> Equation (3) from https://arxiv.org/abs/2112.00384
    """
    commit_loss = F.mse_loss(
                        input=quantizer_input,
                        target=quantizer_output.detach(),
                        reduction='mean',
                    )
    return commit_loss * wt_commit


def mse_commit_loss_cell(
        quantizer_input_cell: torch.Tensor,
        quantizer_output_cell: torch.Tensor,
        wt_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Cell-branch commit loss for the dual VQ architecture (VQNiche_Dual).

    Thin wrapper around `mse_commit_loss` that accepts the suffixed kwarg
    names `quantizer_input_cell` / `quantizer_output_cell` so that the
    loss-dispatcher's key-based lookup can register it independently of
    the niche-branch commit loss without colliding on the legacy keys.

    The two commit-loss branches in VQNiche_Dual are *disjoint*:
        cell:  pulls z_mlp toward z_q_cell  (per-cell features)
        niche: pulls z_gnn toward z_q_niche (post-aggregation features)
    """
    return mse_commit_loss(
        quantizer_input=quantizer_input_cell,
        quantizer_output=quantizer_output_cell,
        wt_commit=wt_commit,
    )


def mse_commit_loss_niche(
        quantizer_input_niche: torch.Tensor,
        quantizer_output_niche: torch.Tensor,
        wt_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Niche-branch commit loss for the dual VQ architecture (VQNiche_Dual).
    See `mse_commit_loss_cell` for the rationale.
    """
    return mse_commit_loss(
        quantizer_input=quantizer_input_niche,
        quantizer_output=quantizer_output_niche,
        wt_commit=wt_commit,
    )


def mse_code_loss(
        quantizer_input: torch.Tensor,
        quantizer_output: torch.Tensor,
        wt_code: float = 0.25
    ) -> torch.Tensor:
    """
    Compute the code loss.

    Parameters
    ----------
    quantizer_input: torch.Tensor
        Input to the VQ module.
        Dimensions: (batch_size, hidden_channels)
    quantizer_output: torch.Tensor
        Output from the VQ module.
        Dimensions: (batch_size, hidden_channels)
    wt_code: float
        The scaling factor for the code loss.

    Returns
    -------
    code_loss: torch.Tensor
        The computed code loss.

    Notes:
    -----
    - The quantizer_input is latent node embeddings from the GNN module.
    - The quantizer_output is quantized node embeddings, i.e. the nearest codebook embedding for each node.
    - This loss function freezes the node embeddings and updates the codebook embeddings.
    - Reference --> Equation (3) from https://arxiv.org/abs/2112.00384
    """
    code_loss = F.mse_loss(
                        input=quantizer_output,
                        target=quantizer_input.detach(),
                        reduction='mean',
                    )
    return code_loss * wt_code 