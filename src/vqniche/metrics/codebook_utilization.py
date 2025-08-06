import torch


def compute_codebook_utilization(
        indices: torch.Tensor,
        codebook_size: int,
        separate: bool,
        num_heads: int,
    ) -> float:
    """
    Compute the codebook utilization.
    
    Parameters
    ----------
    - indices: torch.Tensor
        The indices of the node embeddings mapped to codebook embeddings.
    - codebook_size: int
        The size of the codebook.
    - separate: bool
        Whether the codebook is separated per head.
    - num_heads: int
        The number of heads.

    Returns
    -------
    - utilization: float
        The codebook utilization.
    """
    if num_heads == 1 and not separate:
        # Case 1: Single codebook
        assert indices.ndim == 2  # [B, N]
        used = torch.unique(indices)
        utilization = used.numel() / codebook_size

    elif num_heads == 1 and separate:
        # Case 2: One head but separated (embed_ind will have shape [B, N, 1])
        assert indices.ndim == 3 and indices.shape[-1] == 1
        used = torch.unique(indices)
        utilization = used.numel() / codebook_size

    elif num_heads > 1 and not separate:
        # Case 3: Shared codebook across multiple heads, indices shape: [B, N, H]
        assert indices.ndim == 3 and indices.shape[-1] == num_heads
        used = torch.unique(indices)
        utilization = used.numel() / codebook_size

    elif num_heads > 1 and separate:
        # Case 4: Separate codebook per head, indices shape: [H, B, N]
        assert indices.ndim == 3 and indices.shape[0] == num_heads
        used_per_head = [
            torch.unique(indices[h]).numel()
            for h in range(num_heads)
        ]
        utilization = sum(used_per_head) / (num_heads * codebook_size)

    return utilization