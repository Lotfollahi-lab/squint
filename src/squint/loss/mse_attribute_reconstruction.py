"""
MSE attribute reconstruction loss function.
"""

import torch
import torch.nn.functional as F


def mse_attribute_reconstruction_loss(
        pred_attr: torch.Tensor,
        target_attr: torch.Tensor,
        wt_attr_reconstr: float = 0.1
    ) -> torch.Tensor:
    """
    Compute the mean squared error (MSE) between the estimated attributes from the decoder module and the target attributes.

    Parameters
    ----------
    pred_attr: torch.Tensor
        The output from the attribute decoder module.
        Dimensions: (batch_size, num_genes)
    target_attr: torch.Tensor
        The target attributes.
        Dimensions: (batch_size, num_genes)
    wt_attr_reconstr: float
        The scaling factor for the node attribute reconstruction loss.

    Returns
    -------
    mse_attr_reconstr_loss: torch.Tensor
        The computed node attribute reconstruction loss.

    Notes
    -----
    In VQGraph, `pred_attr` is the output from the Linear attribute decoder module (after vector quantization) and `target_attr` is the log-transformed gene expression values.
    """
    mse_attr_reconstr_loss = F.mse_loss(
                                input=pred_attr,
                                target=torch.log1p(target_attr),
                                reduction='mean',
                            )
    return mse_attr_reconstr_loss * wt_attr_reconstr 