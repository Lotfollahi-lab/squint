import torch
import torch.nn.functional as F


def cosine_similarity(
        embeddings: torch.Tensor,
        prefix: str
    ) -> dict:
    """
    Compute pairwise similarity statistics for all embeddings.

    Returns
    -------
    - similarity_stats: dict
        Dictionary containing mean and std of pairwise cosine similarities for different embeddings
    """
    N = embeddings.size(0)
    # Normalize embeddings for cosine similarity
    normalized = F.normalize(embeddings, p=2, dim=1)

    # Initialize storage for upper triangle similarities
    n_pairs = (N * (N - 1)) // 2
    similarities = torch.empty(n_pairs, device=embeddings.device)

    # Compute only upper triangle elements
    idx = 0
    for i in range(N-1):
        # Compute similarity between embedding i and all j > i
        sims = torch.matmul(normalized[i:i+1], normalized[i+1:].t())
        similarities[idx:idx+N-i-1] = sims[0]
        idx += N-i-1

    return {
        f'{prefix}_mean': similarities.mean().item(),
    }