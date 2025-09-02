import torch


def mask_token_regularization(
        mask_token: torch.Tensor,
        wt_mask_token_regularization: float = 1.0
    ) -> torch.Tensor:
    """
    Compute the mask token regularization loss.
    
    Parameters
    ----------
    mask_token: torch.Tensor
        The mask token.
    wt_mask_token_regularization: float
        The scaling factor for the mask token regularization loss.
        
    Returns
    -------
    mask_token_regularization_loss: torch.Tensor
        The computed mask token regularization loss.
    """
    mask_norm = torch.norm(
                    input=mask_token,
                    p=2,
                    dim=-1,
                )
    return wt_mask_token_regularization * mask_norm