import numpy as np
from sklearn.metrics import accuracy_score as sklearn_accuracy_score

import torch


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
        Dimensions: (batch, num_classes)
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
