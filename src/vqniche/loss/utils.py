import torch
import torch.nn.functional as F


def compute_dispersion(
        input: torch.Tensor,
        num_out_features: int,
        theta: float = 1.0,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
    """
    Compute the dispersion of batch-ids for the negative binomial distribution.

    Parameters
    ----------
    input: torch.Tensor
        The input tensor.
        Dimensions: (batch_size, )
    num_out_features: int
        The number of output features of the dispersion.
    theta: float
        The scaling factor for weight initialization.
    device: torch.device
        The device to use for computation.

    Returns
    -------
    dispersion: torch.Tensor
        The computed dispersion.
        Dimensions: (batch_size, num_out_features)

    Notes
    -----
    - dispersion is computed as the exponential of the linear transformation of the one-hot encoding of the batch-ids.
    - `num_out_features` is the number of genes.
    - the 1D shape of `batch_ids` relies on the assumption that batch-ids are same for all the genes of a cell.
    """
    # Shape of one_hot currently: (batch_size, num_classes)
    one_hot = F.one_hot(
                    input,
                    num_classes=-1 # automatically infer the number of classes
                ).float().to(device)

    # If `input` is of shape (batch_size, *), then `one_hot` will be of shape (batch_size, *, num_classes)
    num_classes = one_hot.shape[-1]

    weight = torch.ones(
                    size=(num_out_features, num_classes),
                    dtype=torch.float32,
                    device=device
                ) * theta

    # Shape of F.linear: (batch_size, num_classes) x (num_out_features, num_classes).t() -> (batch_size, num_out_features)
    dispersions = F.linear(
        input = one_hot, # (batch_size, num_classes)
        weight=weight, # (num_out_features, num_classes)
    )

    # Shape of dispersions: (batch_size, num_out_features)
    return torch.exp(dispersions).to(device)