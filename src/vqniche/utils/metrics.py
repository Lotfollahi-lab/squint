import numpy as np
from sklearn.metrics import accuracy_score as sklearn_accuracy_score

import torch
import torch.nn.functional as F

def accuracy_score(
        unnormalized_logits: torch.Tensor,
        one_hot_labels: torch.Tensor
    ) -> float:
    """
    Compute the accuracy score for a given set of unnormalized logits and labels.

    Parameters
    ----------
    - logits: torch.Tensor
        The unnormalized logits.
        Dimensions: (batch_size, num_classes)
    - labels: torch.Tensor
        The true labels.
        Dimensions: (batch_size, num_classes)

    Returns
    -------
    - accuracy: float
        The accuracy score.
    """
    # compute the predicted class probabilities (normalized logits)
    normalized_logits = unnormalized_logits.softmax(dim=-1)

    # convert to predicted class indices in a numpy array
    predicted_labels = np.argmax(
                            normalized_logits.detach().cpu().numpy(),
                            axis=1,
                        )

    # convert to true class indices in a numpy array
    true_labels = np.argmax(
                            one_hot_labels.detach().cpu().numpy(),
                            axis=1,
                        )

    # compute the accuracy score
    accuracy = sklearn_accuracy_score(
                    y_true=true_labels,
                    y_pred=predicted_labels,
                )

    return accuracy


@staticmethod
def get_similarity_stats(
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
