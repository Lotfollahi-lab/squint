from typing import Literal

import torch
import torch.nn.functional as F

from ..utils.type_conversions import edge_index_to_adjacency_tensor
from ..utils.vqgraph_helpers import l2norm
from ..utils.loss_utils import aggregate_1hop_neighbor_features
from ..utils.adjacency_reconstruction import reconstruct_adjacency_matrix


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
    In VQGraph, `pred_attr` is the output from the Linear attribute decoder module (after vector quantization) and `target_attr` is the log-transformed gene expression values.
    """
    mse_attr_reconstr_loss = F.mse_loss(
                                input=pred_attr,
                                target=torch.log1p(target_attr),
                                reduction='mean',
                            )
    return mse_attr_reconstr_loss * wt_attr_reconstr


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


def mse_adjacency_reconstruction(
        pred_adj: torch.Tensor,
        batch_edge_index: torch.Tensor,
        batch_input_id: torch.Tensor,
        batch_nid: torch.Tensor,
        total_num_nodes: int | None = None,
        adj_reconstr_kwargs: dict = {},
        wt_adj_reconstr: float = 0.1
    ) -> torch.Tensor:
    """
    Compute the mean squared error (MSE) between the estimated adjacency from the decoder module and the original adjacency.

    Parameters
    ----------
    pred_adj: torch.Tensor
        The output from the adjacency decoder module.
        Dimensions: (batch_size + num_sampled_nodes, decoder_embedding_dim)
    batch_edge_index: torch.Tensor
        The edge index of the batch with respect to local node IDs of seed nodes.
        Dimensions: (2, num_edges_in_batch)
    batch_input_id: torch.Tensor
        The global node IDs of the seed nodes of the batch with respect to the entire graph.
        Dimensions: (batch_size,)
    batch_nid: torch.Tensor
        The global node IDs of the seed and all sampled nodes in the batch.
        Dimensions: (batch_size + num_sampled_nodes,)
    total_num_nodes: int | None
        The total number of nodes in the graph.
        If None, the maximum node ID in edge_index will be used.
    adj_reconstr_kwargs: dict
        The keyword arguments for the adjacency reconstruction method.
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
    # i.e. [[0, 1], [1, 2]] means that the 0-th node in the batch is connected to the 1-st node in the batch...
    # so we convert these local node IDs to global node IDs to get that the 0-th node in the batch is the i-th node in the global graph...
    global_edge_index = batch_nid[batch_edge_index]

    # total_num_nodes is the total number of nodes in the graph. 
    # by including total_num_nodes, edge_index_to_adjacency_tensor returns a tensor of shape (total_num_batch, total_num_nodes) where total_num_batch is the number of seed nodes in the batch + all the number of sampled neighbors and total_num_nodes is the total number of nodes in the graph.
    # then, we subset the adjacency tensor to only include the seed nodes in the current batch
    # global_adj is a tensor of shape (batch_size, total_num_nodes) capturing the adjacency information of the seed nodes in the current batch with all nodes in the graph
    # this ensures that computing the loss batch-wise means chunking the adjacency matrix into sets of rows and is thus equivalent to computing the loss on the entire graph
    global_adj = edge_index_to_adjacency_tensor(
                                edge_index=global_edge_index,
                                max_num_nodes=total_num_nodes
                            )[batch_input_id, :]

    # quantize the predicted adjacency matrix coming from the decoder
    # then, subset the quantized adjacency matrix to only include the nodes in the current batch
    adj_reconstr = reconstruct_adjacency_matrix(
                        h_adj=pred_adj,
                        **adj_reconstr_kwargs,
                    )[:batch_input_id.shape[0], :].to(global_adj.device)
    
    # If total_num_nodes is provided and is greater than the current width, pad with zeros
    if total_num_nodes is not None and adj_reconstr.shape[1] < total_num_nodes:
        padding_size = total_num_nodes - adj_reconstr.shape[1]
        adj_reconstr = torch.nn.functional.pad(
            adj_reconstr, 
            pad=(0, padding_size, 0, 0),  # pad only the right side (columns)
            mode='constant', 
            value=0
        )

    # compute the mean root squared error between the quantized adjacency matrix and the original adjacency matrix
    mse_adj_reconstr_loss = torch.sqrt(
                                F.mse_loss(
                                    input=adj_reconstr,
                                    target=global_adj,
                                    reduction='mean'
                                )
                            )

    return mse_adj_reconstr_loss * wt_adj_reconstr


def mse_joint_code_commit_loss(
        quantizer_input: torch.Tensor,
        quantizer_output: torch.Tensor,
        wt_joint_code_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Computes the total codebook loss defined as the sum of the commit loss and code loss.

    Parameters
    ----------
    quantizer_input: torch.Tensor
        Input to the VQ module.
        Dimensions: (batch_size, hidden_channels)
    quantizer_output: torch.Tensor
        Output from the VQ module.
        Dimensions: (batch_size, hidden_channels)
    wt_joint_code_commit: float
        The scaling factor for the total codebook loss.

    Returns
    -------
    joint_code_commit_loss: torch.Tensor
        The computed total codebook loss.

    Notes
    -----
    - The quantizer_input is latent node embeddings from the GNN module.
    - The quantizer_output is quantized node embeddings, i.e. the nearest codebook embedding for each node.
    - Reference: vqgraph --> https://github.com/YangLing0818/VQGraph/blob/main/vq.py
    """
    mse_joint_code_commit_loss = F.mse_loss(
                                    input=quantizer_output.detach(),
                                    target=quantizer_input,
                                    reduction='mean',
                                )
    return mse_joint_code_commit_loss * wt_joint_code_commit


def mse_commit_loss(
        quantizer_input: torch.Tensor,
        quantizer_output: torch.Tensor,
        wt_commit: float = 0.25
    ) -> torch.Tensor:
    """
    Compute the commit loss.

    Parameters
    ----------
    quantizer_input: torch.Tensor
        Input to the VQ module.
        Dimensions: (batch_size, hidden_channels)
    quantizer_output: torch.Tensor
        Output from the VQ module.
        Dimensions: (batch_size, hidden_channels)
    wt_commit: float
        The scaling factor for the commitment loss.

    Returns
    -------
    commit_loss: torch.Tensor
        The computed commit loss.

    Notes:
    -----
    - The quantizer_input is latent node embeddings from the GNN module.
    - The quantizer_output is quantized node embeddings, i.e. the nearest codebook embedding for each node.
    - This loss function freezes the codebook embeddings and updates the node embeddings.
    - Reference --> Equation (3) from https://arxiv.org/abs/2112.00384
    """
    commit_loss = F.mse_loss(
                        input=quantizer_input,
                        target=quantizer_output.detach(),
                        reduction='mean',
                    )
    return commit_loss * wt_commit


def mse_code_loss(
        quantizer_input: torch.Tensor,
        quantizer_output: torch.Tensor,
        wt_code: float = 0.25
    ) -> torch.Tensor:
    """
    Compute the code loss.

    Parameters
    ----------
    quantizer_input: torch.Tensor
        Input to the VQ module.
        Dimensions: (batch_size, hidden_channels)
    quantizer_output: torch.Tensor
        Output from the VQ module.
        Dimensions: (batch_size, hidden_channels)
    wt_code: float
        The scaling factor for the code loss.

    Returns
    -------
    code_loss: torch.Tensor
        The computed code loss.

    Notes:
    -----
    - The quantizer_input is latent node embeddings from the GNN module.
    - The quantizer_output is quantized node embeddings, i.e. the nearest codebook embedding for each node.
    - This loss function freezes the node embeddings and updates the codebook embeddings.
    - Reference --> Equation (3) from https://arxiv.org/abs/2112.00384
    """
    code_loss = F.mse_loss(
                        input=quantizer_output,
                        target=quantizer_input.detach(),
                        reduction='mean',
                    )
    return code_loss * wt_code


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
    normed_codes = l2norm(codebook_embeddings)
    cosine_sim = torch.einsum(
        "h i d, h j d -> h i j",
        normed_codes,
        normed_codes
        )

    codebook_orthogonal_regularization_loss = (cosine_sim**2).sum() / (h * n**2) - (1 / n)

    return codebook_orthogonal_regularization_loss * wt_codebook_orthogonal_regularization