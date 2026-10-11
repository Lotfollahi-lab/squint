import torch


def safe_int_conversion(value):
    """Safely convert a value to int, handling both scalars and tensors"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        else:
            # If multiple values, check they are all the same and take the first
            unique_values = torch.unique(value)
            if len(unique_values) == 1:
                return int(unique_values.item())
            else:
                raise ValueError(f"Expected all values to be the same, but got: {unique_values}")
    else:
        return int(value)