"""
Codebook regularization loss functions.
"""

import torch
import torch.nn.functional as F


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