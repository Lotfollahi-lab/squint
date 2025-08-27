import torch
import torch.nn.functional as F


def set_mask_ratio(
        epoch: int,
        base_ratio: float = 0.0,
        final_ratio: float = 0.6,
        warmup_epochs: int = 10
    ) -> float:
    """
    Set the mask ratio for the current epoch.

    Parameters:
    - epoch: int
        Current epoch
    - base_ratio: float
        Base mask ratio
    - final_ratio: float
        Final mask ratio
    - warmup_epochs: int
        Number of epochs to linearly increase the mask ratio

    Returns:
    -------
    - mask_ratio: float
        Mask ratio
        
    Notes:
    ------
    - The mask ratio is clamped to 0.0 and 1.0 to avoid overflow. 0.0 indicates that no source nodes are masked. 1.0 indicates that all source nodes are masked.
    - If warmup_epochs is 0, the mask ratio is always set to a constant value of `final_ratio`.
    """
    # if warmup_epochs is 0, return final_ratio
    if warmup_epochs == 0:
        return final_ratio
    
    # otherwise, linearly increase mask ratio from base_ratio to final_ratio over warmup_epochs
    mask_ratio = float(base_ratio + final_ratio * min(epoch / warmup_epochs, 1.0))
    
    # clamp mask ratio to 0.0 and 1.0
    mask_ratio = max(mask_ratio, 0.0)
    mask_ratio = min(mask_ratio, 1.0)
    
    return mask_ratio


def set_mask_indices(
        N: int,
        batch_size: int,
        mask_ratio: float,
        deterministic: bool = False
    ) -> torch.LongTensor:
    """
    Mask a subset of the first `batch_size` nodes (source nodes of the batch).
    
    Parameters:
    ----------
    - N: int
        Total number of nodes
    - batch_size: int
        Number of source nodes of the batch to consider for masking
    - mask_ratio: float
        Fraction of source nodes of the batch to mask
    - deterministic: bool
        If True, uses deterministic stride-based masking.
        If False, uses random masking (default).

    Returns:
    -------
    - mask_idx: torch.LongTensor
        Long tensor of shape [N] with 1 for masked nodes and 0 for unmasked nodes
    """
    assert mask_ratio >= 0.0 and mask_ratio <= 1.0, "Mask ratio must be between 0 and 1"
    
    # initialize mask indices to 0 for all nodes
    mask_idx = torch.zeros(N, dtype=torch.long)

    # if mask ratio is 0, return all nodes as not masked
    if mask_ratio == 0.0:
        pass
    
    # if mask ratio is 1, return all source nodes as masked
    elif mask_ratio == 1.0:
        mask_idx[:batch_size] = 1
    
    # if mask ratio is between 0 and 1, sample a subset of source nodes to mask
    elif mask_ratio > 0.0 and mask_ratio < 1.0:
        k = max(1, int(round(mask_ratio * batch_size)))
    
        if deterministic:
            # Calculate stride to evenly space the masked source nodes
            stride = max(1, batch_size // k)
            # Take evenly spaced indices up to k source nodes
            idx = torch.arange(0, min(k * stride, batch_size), stride)
        else:
            # Randomly sample k source nodes
            idx = torch.randperm(batch_size)[:k]
            
        mask_idx[idx] = 1

    return mask_idx


@torch.no_grad()
def print_masked_input_diversity_stats(
        x_in: torch.Tensor,
        batch_size: int,
        mask_idx: torch.Tensor | None = None,
        round_decimals: int = 6
    ) -> dict:
    """
    Computes and prints quick diagnostics for input diversity among masked nodes.
    Low values (or high cosine) indicate collapse.

    Parameters:
    ----------
    - x_in: torch.Tensor
        Input tensor of shape [N, F]
    - batch_size: int
        Number of center nodes to consider for masking
    - mask_idx: torch.Tensor | None
        Boolean mask tensor of shape [N] or None
    - round_decimals: int
        Number of decimal places to round the input tensor to (default: 6)

    Returns:
    -------
    - stats: dict
        Dictionary containing the following statistics:
        - masked_count: int
        - mean_feat_std: float
        - unique_row_frac: float
        - mean_pair_cos: float
        - p95_pair_cos: float
    """
    N = x_in.size(0)
    if mask_idx is None:
        mask = torch.zeros(N, dtype=torch.bool, device=x_in.device)
        mask[:batch_size] = True           # NeighborLoader: centers are first `batch_size`
    else:
        mask = mask_idx if mask_idx.dtype == torch.bool else torch.zeros(
            N, dtype=torch.bool, device=x_in.device).scatter_(0, mask_idx.to(x_in.device), True)

    X = x_in[mask]                         # [M, F] masked inputs
    M = X.size(0)
    if M <= 1:
        return {"masked_count": int(M), "mean_feat_std": float('nan'),
                "unique_row_frac": 1.0, "mean_pair_cos": float('nan'),
                "p95_pair_cos": float('nan')}

    # 1) Per-feature std across masked nodes (0 => identical per feature)
    per_feat_std = X.std(dim=0)
    mean_feat_std = per_feat_std.mean().item()

    # 2) Unique rows fraction (≈0 if all rows equal)
    Xr = torch.round(X * (10**round_decimals)) / (10**round_decimals)
    unique_rows = torch.unique(Xr, dim=0).size(0)
    unique_row_frac = float(unique_rows) / float(M)

    # 3) Pairwise cosine similarity among masked rows (≈1 if identical)
    Xn = F.normalize(X, dim=1)
    cos = Xn @ Xn.T
    off = cos[~torch.eye(M, dtype=torch.bool, device=X.device)]
    mean_pair_cos = off.mean().item()
    p95_pair_cos = off.quantile(0.95).item()

    stats = {
        "masked_count": int(M),
        "mean_feat_std": mean_feat_std,
        "unique_row_frac": unique_row_frac,
        "mean_pair_cos": mean_pair_cos,
        "p95_pair_cos": p95_pair_cos,
    }
    print(stats)

    return stats