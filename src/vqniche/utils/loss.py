import torch
import torch.functional as F


def cross_entropy_loss(logits: torch.Tensor,
                       labels: torch.Tensor,
                       reduction: str = "none") -> torch.Tensor:
    """
    Compute the cross-entropy loss for multiclass classification.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized logits.
    labels : torch.Tensor
        Ground truth class indices or class probabilities.

    Returns
    -------
    torch.Tensor
        The computed cross-entropy loss.

    Notes
    -----
    Cross entropy loss can take predicted class probabilities as input. But we are using the unnormalized logits as input for numerical stability and because the cross_entropy function in PyTorch automatically applies the softmax function to the logits.
    """
    return F.cross_entropy(input=logits,
                           target=labels,
                           reduction=reduction)