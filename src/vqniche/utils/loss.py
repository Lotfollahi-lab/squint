import torch
import torch.nn.functional as F
from vqniche.utils.type_conversions import edge_index_to_adjacency_tensor

from vqniche.utils.vqgraph_helpers import l2norm


def cross_entropy_loss(logits: torch.Tensor,
                       labels: torch.Tensor,
                       reduction: str = "mean"
    ) -> torch.Tensor:
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
    return F.cross_entropy(
        input=logits,
        target=labels,
        reduction=reduction
    )


def vqgraph_attribute_reconstruction(
    h_pre_vq_conv: torch.Tensor,
    h_node: torch.Tensor,
    scaling_node_gamma: float = 0.001
    ) -> torch.Tensor:
    """
    Compute the node attribute reconstruction loss for VQGraph.

    Parameters
    ----------
    h_pre_vq_conv: torch.Tensor
        The latent node embedding obtained from the pre-VQ graph convolution layer(s).
    h_node: torch.Tensor
        The quantized node embedding obtained from a Linear Decoder layer on the output of the VQ layer
    scaling_node_gamma: float
        The scaling factor for the node attribute reconstruction loss.

    Returns
    -------
    torch.Tensor
        The computed node attribute reconstruction loss.
    """

    return F.mse_loss(
        input=h_node,
        target=h_pre_vq_conv,
        reduction='mean'
    ) * scaling_node_gamma


def vqgraph_adjacency_reconstruction(
    batch_edge_index: torch.Tensor,
    h_edge: torch.Tensor,
    scaling_edge_gamma: float = 0.03
    ) -> torch.Tensor:
    """
    Compute the adjacency reconstruction loss for VQGraph.

    Parameters
    ----------
    batch_edge_index: torch.Tensor
        The edge index of the batch
    h_edge: torch.Tensor
        The quantized edge embedding obtained from a Linear Decoder layer on the output of the VQ layer
    scaling_edge_gamma: float
        The scaling factor for the adjacency reconstruction loss.

    Returns
    -------
    torch.Tensor
        The computed adjacency reconstruction loss.
    """
    batch_adjacency_matrix = edge_index_to_adjacency_tensor(
        edge_index=batch_edge_index,
        num_nodes=h_edge.shape[0],
        device=batch_edge_index.device
    )

    adj_quantized = torch.matmul(h_edge, h_edge.t())
    adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())
    adj_quantized = adj_quantized.to(batch_adjacency_matrix.device)

    return torch.sqrt(F.mse_loss(
        input=adj_quantized,
        target=batch_adjacency_matrix,
        reduction='mean'
    )) * scaling_edge_gamma


def vqgraph_commitment_loss(
    h_pre_vq_conv: torch.Tensor,
    h_vq: torch.Tensor,
    commitment_weight: float = 0.25
    ) -> torch.Tensor:
    """
    Compute the commitment loss for VQGraph.

    Parameters
    ----------
    h_pre_vq_conv: torch.Tensor
        The latent node embedding obtained from the pre-VQ graph convolution layer(s).
    h_vq: torch.Tensor
        The quantized node embedding obtained from a Linear Decoder layer on the output of the VQ layer
    commitment_weight: float
        The scaling factor for the commitment loss.

    Returns
    -------
    torch.Tensor
        The computed commitment loss.
    """
    h_vq = h_pre_vq_conv + (h_vq - h_pre_vq_conv).detach()
    detached_h_vq = h_vq.detach()
    return F.mse_loss(
        input=h_pre_vq_conv,
        target=detached_h_vq,
        reduction='mean'
    ) * commitment_weight


def vqgraph_codebook_loss(
    codebook_embeddings: torch.Tensor,
    codebook_reg_weight: float = 0.001,
    codebook_reg_active_codes_only: bool = False,
    codebook_reg_max_codes: int = None
    ) -> torch.Tensor:
    """
    Compute the codebook loss for VQGraph.

    Parameters
    ----------
    codebook_embeddings: torch.Tensor
        The codebook embeddings.
    codebook_reg_weight: float
        The scaling factor for the codebook loss.
    codebook_reg_active_codes_only: bool
        Whether to only calculate the codebook loss for the active codes.
    codebook_reg_max_codes: int
        The maximum number of codes to use for the codebook loss.

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

    return codebook_loss * codebook_reg_weight