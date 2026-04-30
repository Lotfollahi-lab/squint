from typing import Sequence, Union

import torch


def compute_codebook_utilization(
        indices: torch.Tensor,
        codebook_size: Union[int, Sequence[int]],
        separate: bool,
        num_heads: int,
        num_quantizers: int = 1,
    ) -> float:
    """
    Compute the codebook utilization.

    Parameters
    ----------
    - indices: torch.Tensor
        The indices of the node embeddings mapped to codebook embeddings.
    - codebook_size: int or sequence of ints
        Codebook size. For single-codebook VQ this is a scalar. For multi-
        level VQ (RVQ / ConditionalVQ) it MAY be a sequence of length
        `num_quantizers`, one per level.
    - separate: bool
        Whether the codebook is separated per head.
    - num_heads: int
        The number of heads (multi-head VQ).
    - num_quantizers: int
        The number of quantization levels (RVQ / ConditionalVQ). 1 = single
        codebook (legacy behaviour). >1 = multi-level; indices is expected
        to have a trailing dimension of size num_quantizers.

    Returns
    -------
    - utilization: float
        The codebook utilization, in [0, 1].
        For multi-level VQ, this is the mean of per-level utilizations.
    """
    # ------------------------------------------------------------------
    # Multi-level (RVQ / ConditionalVQ): indices has a trailing axis of
    # size num_quantizers. Compute per-level utilization and average.
    # ------------------------------------------------------------------
    if num_quantizers is not None and int(num_quantizers) > 1:
        if isinstance(codebook_size, int):
            sizes = [int(codebook_size)] * int(num_quantizers)
        else:
            sizes = [int(k) for k in codebook_size]
            assert len(sizes) == int(num_quantizers), (
                f"len(codebook_size) {len(sizes)} != num_quantizers {num_quantizers}"
            )
        # indices is expected to have shape (..., num_quantizers).
        # Flatten leading dims; unique-count per level along the last axis.
        idx_flat = indices.reshape(-1, int(num_quantizers))
        per_level = []
        for q in range(int(num_quantizers)):
            used_q = torch.unique(idx_flat[:, q]).numel()
            per_level.append(used_q / sizes[q])
        return float(sum(per_level) / len(per_level))

    # ------------------------------------------------------------------
    # Single-level (legacy paths).  codebook_size must be a scalar here.
    # ------------------------------------------------------------------
    K = int(codebook_size if isinstance(codebook_size, int) else codebook_size[0])

    if num_heads == 1 and not separate:
        # Case 1: Single codebook
        used = torch.unique(indices)
        utilization = used.numel() / K

    elif num_heads == 1 and separate:
        # Case 2: One head but separated (embed_ind will have shape [B, N, 1])
        used = torch.unique(indices)
        utilization = used.numel() / K

    elif num_heads > 1 and not separate:
        # Case 3: Shared codebook across multiple heads, indices shape: [B, N, H]
        used = torch.unique(indices)
        utilization = used.numel() / K

    elif num_heads > 1 and separate:
        # Case 4: Separate codebook per head, indices shape: [H, B, N]
        used_per_head = [
            torch.unique(indices[h]).numel()
            for h in range(num_heads)
        ]
        utilization = sum(used_per_head) / (num_heads * K)

    return float(utilization)
