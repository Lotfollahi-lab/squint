"""
Negative binomial attribute reconstruction loss function.
"""

from typing import Literal

import torch
import torch.nn.functional as F

from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features


def nb_attribute_reconstruction_loss(
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
    # if k_hop_nb_loss == 1:
    #     pred_attr = aggregate_1hop_neighbor_features(
    #                     X=pred_attr,
    #                     edge_index=edge_index,
    #                     return_mean=False,
    #                 )
    #     target_attr = aggregate_1hop_neighbor_features(
    #                     X=target_attr,
    #                     edge_index=edge_index,
    #                     return_mean=False,
    #                 )

    # pred_attr = pred_attr[:batch_size]
    # target_attr = target_attr[:batch_size]

    # NOTE: do NOT detach this term — it is the only place where the gradient
    # of the NB log-likelihood w.r.t. the predicted mean (mu) flows through the
    # `dispersion * log(theta + mu)` component. Detaching it (previous code)
    # silently dropped a gradient term and biased the loss. This matches the
    # scvi-tools reference implementation of `log_nb_positive`.
    log_theta_mu_eps = torch.log(dispersion + pred_attr + 1e-8)
    log_likelihood_nb = (
        dispersion * (torch.log(dispersion + 1e-8) - log_theta_mu_eps)
        + target_attr * (torch.log(pred_attr + 1e-8) - log_theta_mu_eps)
        + torch.lgamma(target_attr + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(target_attr + 1))

    nb_loss = torch.mean(-log_likelihood_nb.sum(-1))

    return nb_loss * wt_attr_reconstr


def nb_nbr_attribute_reconstruction_loss(
        pred_attr_nbr: torch.Tensor,
        target_attr_nbr: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        dispersion: torch.Tensor,
        k_hop_nb_loss: int = 0,
        wt_attr_reconstr: float = 0.1,
    ) -> torch.Tensor:
    """
    Thin wrapper around `nb_attribute_reconstruction_loss` for the
    neighbourhood branch when `recon_mode='both'`.

    The loss dispatcher extracts tensors from the step's loss_data dict by
    matching keyword names to dict keys.  The neighbourhood targets are stored
    under `pred_attr_nbr` / `target_attr_nbr` to avoid colliding with the
    per-cell keys (`pred_attr` / `target_attr`). This wrapper accepts those
    suffixed names and forwards them to the canonical loss under the expected
    positional names.

    The upstream code (training_step / validation_step) already computes the
    1-hop aggregation before storing the pair, so `k_hop_nb_loss` is always
    passed as 0 by the dispatcher — the aggregation is NOT repeated here.
    """
    return nb_attribute_reconstruction_loss(
        pred_attr=pred_attr_nbr,
        target_attr=target_attr_nbr,
        edge_index=edge_index,
        batch_size=batch_size,
        dispersion=dispersion,
        k_hop_nb_loss=k_hop_nb_loss,
        wt_attr_reconstr=wt_attr_reconstr,
    )