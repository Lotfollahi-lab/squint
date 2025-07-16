"""
Cross-entropy loss function for multiclass classification.
"""

import torch
import torch.nn.functional as F


def cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        wt_cross_entropy: float = 1.0
    ) -> torch.Tensor:
    """
    Compute the cross-entropy loss for multiclass classification.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized logits.
        Dimensions: (batch_size, num_classes)
    labels : torch.Tensor
        Ground truth class indices or class probabilities.
        Dimensions: (batch_size,)
    wt_cross_entropy : float
        The scaling factor for the cross-entropy loss.

    Returns
    -------
    ce_loss: torch.Tensor
        The computed cross-entropy loss weighted by `wt_cross_entropy`.

    Notes
    -----
    Cross entropy loss can take predicted class probabilities as input. But we are using the unnormalized logits as input for numerical stability and because the cross_entropy function in PyTorch automatically applies the softmax function to the logits.
    """
    ce_loss = F.cross_entropy(
                input=logits,
                target=labels,
                reduction='mean',
            )
    return ce_loss * wt_cross_entropy 