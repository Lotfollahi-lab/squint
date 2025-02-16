import torch
import torch.nn.functional as F


def compute_dispersion(
    input_tensor: torch.Tensor,
    n_genes: int,
    num_classes: int = -1,
    theta: float = 1.0,
    device: torch.device = torch.device("cpu")
    ):
    print(f"{input_tensor.shape=}")


    one_hot = F.one_hot(
            input_tensor,
            num_classes=num_classes
        ).float().to(device)
    print(f"{one_hot=}")
    print(f"{one_hot.shape=}")
    n_cells, n_classes = one_hot.shape

    weight = torch.ones(
                    (n_genes, n_classes),
                    dtype=torch.float32,
                    device=device
                ) * theta
    print(f"{weight.shape=}")

    dispersions = F.linear(
        input = one_hot,
        weight=weight,
    )

    return torch.exp(dispersions).to(device)