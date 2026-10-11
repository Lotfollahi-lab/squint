"""
VQGraph-specific codebook loss functions including joint loss and regularization.
"""

import torch
import torch.nn.functional as F


def mse_joint_code_commit_loss(
        quantizer_input: torch.Tensor,
        quantizer_output: torch.Tensor,
        wt_joint_code_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Computes the total codebook loss defined as the sum of the commit loss and code loss.

    Parameters
    ----------
    quantizer_input: torch.Tensor
        Input to the VQ module.
        Dimensions: (batch_size, hidden_channels)
    quantizer_output: torch.Tensor
        Output from the VQ module.
        Dimensions: (batch_size, hidden_channels)
    wt_joint_code_commit: float
        The scaling factor for the total codebook loss.

    Returns
    -------
    joint_code_commit_loss: torch.Tensor
        The computed total codebook loss.

    Notes
    -----
    - The quantizer_input is latent node embeddings from the GNN module.
    - The quantizer_output is quantized node embeddings, i.e. the nearest codebook embedding for each node.
    - Reference: vqgraph --> https://github.com/YangLing0818/VQGraph/blob/main/vq.py
    """
    mse_joint_code_commit_loss = F.mse_loss(
                                    input=quantizer_output.detach(),
                                    target=quantizer_input,
                                    reduction='mean',
                                )
    return mse_joint_code_commit_loss * wt_joint_code_commit


def l2_codebook_orthogonal_regularization_loss(
        codebook_embeddings: torch.Tensor,
        wt_codebook_orthogonal_regularization: float = 0.2,
        codebook_reg_active_codes_only: bool = False,
        codebook_reg_max_codes: int = None
    ) -> torch.Tensor:
    """
    Compute the codebook orthogonal regularization loss for VQGraph.

    Parameters
    ----------
    codebook_embeddings: torch.Tensor
        The codebook embeddings.
        Dimensions: (codebook_size, num_genes)
    wt_codebook_orthogonal_regularization: float
        The scaling factor for the codebook orthogonal regularization loss.
    codebook_reg_active_codes_only: bool
        Whether to only calculate the codebook orthogonal regularization loss for the active codes.
    codebook_reg_max_codes: int
        The maximum number of codes to use for the codebook orthogonal regularization loss.

    Returns
    -------
    codebook_orthogonal_regularization_loss: torch.Tensor
        The computed codebook orthogonal regularization loss.

    """
    if codebook_reg_active_codes_only:
        raise NotImplementedError("Codebook orthogonal regularization loss for active codes only is not implemented.")

    if codebook_reg_max_codes is not None:
        raise NotImplementedError("Codebook orthogonal regularization loss for max codes is not implemented.")

    h, n = codebook_embeddings.shape[:2]
    normed_codes = F.normalize(
                        codebook_embeddings,
                        p=2,
                        dim=-1,
                    )
    cosine_sim = torch.einsum(
        "h i d, h j d -> h i j",
        normed_codes,
        normed_codes
        )

    codebook_orthogonal_regularization_loss = (cosine_sim**2).sum() / (h * n**2) - (1 / n)

    return codebook_orthogonal_regularization_loss * wt_codebook_orthogonal_regularization 