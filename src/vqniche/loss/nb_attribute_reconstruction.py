"""
Negative binomial attribute reconstruction loss function.
"""

from typing import Literal

import torch
import torch.nn.functional as F

from ...utils.loss_utils import aggregate_1hop_neighbor_features


def nb_attribute_reconstruction(
        pred_attr: torch.Tensor,
        target_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        dispersion: torch.Tensor,
        k_hop_nb_loss: Literal[0, 1] = 0,
        wt_attr_reconstr: float = 0.1
    ) -> torch.Tensor:
    """
    Compute the negative binomial (NB) loss between the estimated attributes from the decoder module and the target attributes.

    Parameters
    ----------
    pred_attr: torch.Tensor
        The output from the attribute decoder module.
        Dimensions: (batch_size, num_genes)
    target_attr: torch.Tensor
        The target attributes.
        Dimensions: (batch_size, num_genes)
    edge_index: torch.Tensor
        The edge index of the graph.
        Dimensions: (2, num_edges)
    batch_size: int
        The size of the batch.
    dispersion: torch.Tensor
        The dispersion parameter.
        Dimensions: (num_genes,)
    k_hop_nb_loss: Literal[0, 1]
        The number of hops to consider for the neighbor features. 0 indicates individual node attributes, 1 indicates 1-hop neighbor features.
    wt_attr_reconstr: float
        The scaling factor for the node attribute reconstruction loss.

    Returns
    -------
    nb_loss: torch.Tensor
        The computed negative binomial loss.

    Notes
    -----
    - This implementation sets one dispersion parameter per gene.
    - mu (mean of the NB distribution) is the predicted attribute.
    - theta (dispersion parameter) is a learnable parameter.
    - target_attr is the raw count data.
    - NB loss seeks to estimate the true counts conditioned on the predicted attributes.

    References:
    ----------
    - NicheCompass --> https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/modules/losses.py
    - scvi-tools --> https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py#L205
    """
    if k_hop_nb_loss == 1:
        pred_attr = aggregate_1hop_neighbor_features(
                        X=pred_attr,
                        edge_index=edge_index,
                        return_mean=False,
                    )
        target_attr = aggregate_1hop_neighbor_features(
                        X=target_attr,
                        edge_index=edge_index,
                        return_mean=False,
                    )

    pred_attr = pred_attr[:batch_size]
    target_attr = target_attr[:batch_size]

    log_theta_mu_eps = torch.log(dispersion + pred_attr + 1e-8).detach()
    log_likelihood_nb = (
        dispersion * (torch.log(dispersion + 1e-8) - log_theta_mu_eps)
        + target_attr * (torch.log(pred_attr + 1e-8) - log_theta_mu_eps)
        + torch.lgamma(target_attr + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(target_attr + 1))

    nb_loss = torch.mean(-log_likelihood_nb.sum(-1))

    return nb_loss * wt_attr_reconstr 