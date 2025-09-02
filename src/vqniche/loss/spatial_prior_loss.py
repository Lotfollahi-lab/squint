"""
Spatial prior loss function for the codebook.
"""

import torch
import torch.nn.functional as F


def ce_spatial_prior_loss(
        h_spatial_prior: torch.Tensor,
        indices_one_hot: torch.Tensor,
        wt_spatial_prior: float = 1.0
    ) -> torch.Tensor:
    """
    Compute the spatial prior loss for the codebook.

    Parameters
    ----------
    h_spatial_prior : torch.Tensor
        Unnormalized logits for the spatial prior.
        Dimensions: (batch_size, num_classes)
    indices_one_hot : torch.Tensor
        Ground truth class indices or class probabilities for the spatial prior.
        Dimensions: (batch_size,)
    wt_spatial_prior : float
        The scaling factor for the spatial prior loss.

    Returns
    -------
    sp_loss: torch.Tensor
        The computed cross-entropy loss weighted by `wt_spatial_prior`.

    Notes
    -----
    Spatial prior loss can take predicted class probabilities as input. But we are using the unnormalized logits as input for numerical stability and because the cross_entropy function in PyTorch automatically applies the softmax function to the logits.
    """
    sp_loss = F.cross_entropy(
                input=h_spatial_prior,
                target=indices_one_hot.detach().to(torch.float32),
                reduction='mean',
            )
    return sp_loss * wt_spatial_prior 