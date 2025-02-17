import torch
import torch.nn.functional as F
from typing import Literal, Optional

from vqniche.utils.type_conversions import edge_index_to_adjacency_tensor
from vqniche.utils.vqgraph_helpers import l2norm
from vqniche.utils.loss_utils import compute_dispersion
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial


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


def mse_attribute_reconstruction(
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
    In VQGraph, `pred_attr` is the output from the Linear attribute decoder module (after vector quantization) and `target_attr` is the node embedding from the pre-VQ graph convolution layer(s).
    """
    mse_attr_reconstr_loss = F.mse_loss(
                                input=target_attr,
                                target=pred_attr,
                                reduction='mean',
                            )
    return mse_attr_reconstr_loss * wt_attr_reconstr


def nb_attribute_reconstruction(
        pred_attr: torch.Tensor,
        target_attr: torch.Tensor,
        dispersion: Optional[torch.Tensor] = None,
        distribution: Literal['zinb', 'nb'] = 'nb',
        dispersion_theta: float = 1.0,
        wt_attr_reconstr: float = 0.1,
    ) -> torch.Tensor:
    """
    Compute the negative binomial loss.

    Parameters
    ----------
    pred_attr: torch.Tensor
        The output from the attribute decoder module.
        Dimensions: (batch_size, num_genes)
    target_attr: torch.Tensor
        The target attributes.
        Dimensions: (batch_size, num_genes)
    dispersion: torch.Tensor
        The dispersion parameter for the negative binomial distribution. Can be `None` if not provided.
    distribution: str
        The distribution to use for the negative binomial loss. Can be 'nb' or 'zinb'.
    dispersion_theta: float
        The `theta` parameter to compute dispersion of batch-ids for the negative binomial distribution.
    wt_attr_reconstr: float
        The scaling factor for the node attribute reconstruction loss.

    Returns
    -------
    nb_loss: torch.Tensor
        The computed node attribute reconstruction loss.

    Notes
    -----
    - If we set `target_attr` to be the raw count data, we could repurpose this to be used for a Count Decoder in the future.
    - Otherwise, with real-valued tensors, this returns NaNs.
    """
    if dispersion is None:
        # TODO: Replace with batch_ids from the dataloader
        batch_ids = torch.ones(
                        pred_attr.shape[0], # batch_size
                        dtype=torch.long,
                        device=pred_attr.device
                    )

        dispersion = compute_dispersion(
            input=batch_ids,
            num_out_features=pred_attr.shape[1], # num_genes
            theta=dispersion_theta,
            device=pred_attr.device
        )
    else:
        dispersion = dispersion.expand(
                        pred_attr.shape[0], -1
                    ).to(pred_attr.device) # (batch_size, num_genes)

    if distribution == 'zinb':
        nb_distribution = ZeroInflatedNegativeBinomial(
                            mu=pred_attr,
                            theta=dispersion
                            )
    elif distribution == 'nb':
        nb_distribution = NegativeBinomial(
                            mu=pred_attr,
                            theta=dispersion
                            )
    nb_loss = -nb_distribution.log_prob(
                                target_attr
                            ).sum(dim=-1).mean()

    return nb_loss * wt_attr_reconstr


def mse_adjacency_reconstruction(
        pred_adj: torch.Tensor,
        batch_edge_index: torch.Tensor,
        wt_adj_reconstr: float = 0.1
    ) -> torch.Tensor:
    """
    Compute the mean squared error (MSE) between the estimated adjacency from the decoder module and the original adjacency.

    Parameters
    ----------
    pred_adj: torch.Tensor
        The output from the adjacency decoder module.
        Dimensions: (batch_size, num_genes)
    batch_edge_index: torch.Tensor
        The edge index of the batch
        Dimensions: (2, num_edges)
    wt_adj_reconstr: float
        The scaling factor for the adjacency reconstruction loss.

    Returns
    -------
    mse_adj_reconstr_loss: torch.Tensor
        The computed adjacency reconstruction loss.

    Notes
    -----
    In VQGraph, `pred_adj` is the output from the Linear adjacency decoder module (after vector quantization) and `batch_edge_index` is the edge index of the batch.
    """
    # TODO: fix using global node ID
    batch_adjacency_matrix = edge_index_to_adjacency_tensor(
        edge_index=batch_edge_index,
        num_nodes=pred_adj.shape[0],
        device=batch_edge_index.device
    )

    adj_quantized = torch.matmul(pred_adj, pred_adj.t())
    adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())
    adj_quantized = adj_quantized.to(batch_adjacency_matrix.device)

    mse_adj_reconstr_loss = torch.sqrt(F.mse_loss(
        input=adj_quantized,
        target=batch_adjacency_matrix,
        reduction='mean'
    ))

    return mse_adj_reconstr_loss * wt_adj_reconstr


def mse_commitment_loss(
        pred_commit: torch.Tensor,
        target_commit: torch.Tensor,
        wt_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Compute the commitment loss for the VQ layer using a straight-through estimator.

    Parameters
    ----------
    pred_commit: torch.Tensor
        The latent node embedding obtained from the pre-VQ graph convolution layer(s).
        Dimensions: (batch_size, num_genes)
    target_commit: torch.Tensor
        The quantized node embedding obtained from a Linear Decoder layer on the output of the VQ layer.
        Dimensions: (batch_size, num_genes)
    wt_commit: float
        The scaling factor for the commitment loss.

    Returns
    -------
    mse_commit_loss: torch.Tensor
        The computed commitment loss.
    """
    target_commit = pred_commit + (target_commit - pred_commit).detach()
    detached_target = target_commit.detach()

    mse_commit_loss = F.mse_loss(
                        input=pred_commit,
                        target=detached_target,
                        reduction='mean',
                    )
    return mse_commit_loss * wt_commit


def l2_codebook_loss(
        codebook_embeddings: torch.Tensor,
        wt_codebook: float = 0.2,
        codebook_reg_active_codes_only: bool = False,
        codebook_reg_max_codes: int = None
    ) -> torch.Tensor:
    """
    Compute the codebook loss for VQGraph.

    Parameters
    ----------
    codebook_embeddings: torch.Tensor
        The codebook embeddings.
        Dimensions: (codebook_size, num_genes)
    codebook_reg_weight: float
        The scaling factor for the codebook loss.
    codebook_reg_active_codes_only: bool
        Whether to only calculate the codebook loss for the active codes.
    codebook_reg_max_codes: int
        The maximum number of codes to use for the codebook loss.

    Returns
    -------
    codebook_loss: torch.Tensor
        The computed codebook loss.

    Notes:
    -----
    - Source --> Equation (3) from https://arxiv.org/abs/2112.00384
    """
    if codebook_reg_active_codes_only:
        raise NotImplementedError("Codebook loss for active codes only is not implemented.")

    if codebook_reg_max_codes is not None:
        raise NotImplementedError("Codebook loss for max codes is not implemented.")

    h, n = codebook_embeddings.shape[:2]
    normed_codes = l2norm(codebook_embeddings)
    cosine_sim = torch.einsum(
        "h i d, h j d -> h i j",
        normed_codes,
        normed_codes
        )

    codebook_loss = (cosine_sim**2).sum() / (h * n**2) - (1 / n)

    return codebook_loss * wt_codebook