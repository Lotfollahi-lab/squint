import torch
import torch.nn.functional as F

from ..utils.type_conversions import edge_index_to_adjacency_tensor
from ..utils.vqgraph_helpers import l2norm


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
                                input=pred_attr,
                                target=target_attr,
                                reduction='mean',
                            )
    return mse_attr_reconstr_loss * wt_attr_reconstr


def mse_adjacency_reconstruction(
        pred_adj: torch.Tensor,
        batch_edge_index: torch.Tensor,
        batch_input_id: torch.Tensor,
        batch_nid: torch.Tensor,
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
        The edge index of the batch with respect to local node IDs of seed nodes.
        Dimensions: (2, num_edges_in_batch)
    batch_input_id: torch.Tensor
        The global node IDs of the seed nodes of the batch with respect to the entire graph.
        Dimensions: (batch_size,)
    batch_nid: torch.Tensor
        The global node IDs of the seed and all sampled nodes in the batch.
    wt_adj_reconstr: float
        The scaling factor for the adjacency reconstruction loss.

    Returns
    -------
    mse_adj_reconstr_loss: torch.Tensor
        The computed adjacency reconstruction loss.

    Notes
    -----
    - In VQGraph, `pred_adj` is the output from the adjacency decoder module (after vector quantization).
    - this implementation assumes that sampling and mini-batching is used in the dataloader.
    """
    # sampling-dataloaders (e.g. NeighborLoader) use local node IDs by default in the batch_edge_index.
    # we need to convert these local node IDs to global node IDs to subset the adjacency tensor
    global_edge_index = batch_nid[batch_edge_index]

    # edge_index_to_adjacency_tensor returns a tensor of shape (num_nodes, num_nodes) where num_nodes is the maximum node ID in global_edge_index
    # we subset the adjacency tensor to only include the nodes in the current batch
    global_batch_adj = edge_index_to_adjacency_tensor(
                                edge_index=global_edge_index,
                            )[batch_input_id, :][:, batch_input_id]

    # quantize the predicted adjacency matrix coming from the decoder
    # then, subset the quantized adjacency matrix to only include the nodes in the current batch
    adj_quantized = torch.matmul(pred_adj.detach(), pred_adj.detach().t())
    adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min() + 1e-8)
    adj_quantized = adj_quantized.to(global_batch_adj.device)

    # compute the mean root squared error between the quantized adjacency matrix and the original adjacency matrix
    mse_adj_reconstr_loss = torch.sqrt(
                                F.mse_loss(
                                    input=adj_quantized,
                                    target=global_batch_adj,
                                    reduction='mean'
                                )
                            )

    return mse_adj_reconstr_loss * wt_adj_reconstr


def mse_total_codebook_loss(
        pred_commit: torch.Tensor,
        target_commit: torch.Tensor,
        wt_total_codebook: float = 0.25
    ) -> torch.Tensor:
    """
    Computes the total codebook loss defined as the sum of the commit loss and code loss for the VQGraph encoder as in the original VQGraph implementation.

    Parameters
    ----------
    pred_commit: torch.Tensor
        The latent node embedding obtained from the pre-VQ graph convolution layer(s).
        Dimensions: (batch_size, num_genes)
    target_commit: torch.Tensor
        The quantized node embedding obtained from a Linear Decoder layer on the output of the VQ layer.
        Dimensions: (batch_size, num_genes)
    wt_total_codebook: float
        The scaling factor for the total codebook loss.

    Returns
    -------
    total_codebook_loss: torch.Tensor
        The computed total codebook loss.

    Notes
    -----
    - The pred_commit for VQGraph is the output from the pre-VQ graph convolution layer(s).
    - The target_commit for VQGraph is the output from the VQ layer (i.e. the quantized node embedding obtained from the codebook).
    """
    detached_target = target_commit.detach()
    mse_total_codebook_loss = F.mse_loss(
                        input=pred_commit,
                        target=detached_target,
                        reduction='mean',
                    )
    return mse_total_codebook_loss * wt_total_codebook


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

    Notes:
    -----
    - Source --> Equation (3) from https://arxiv.org/abs/2112.00384
    """
    if codebook_reg_active_codes_only:
        raise NotImplementedError("Codebook orthogonal regularization loss for active codes only is not implemented.")

    if codebook_reg_max_codes is not None:
        raise NotImplementedError("Codebook orthogonal regularization loss for max codes is not implemented.")

    h, n = codebook_embeddings.shape[:2]
    normed_codes = l2norm(codebook_embeddings)
    cosine_sim = torch.einsum(
        "h i d, h j d -> h i j",
        normed_codes,
        normed_codes
        )

    codebook_orthogonal_regularization_loss = (cosine_sim**2).sum() / (h * n**2) - (1 / n)

    return codebook_orthogonal_regularization_loss * wt_codebook_orthogonal_regularization