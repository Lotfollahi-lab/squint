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

    Notes
    -----
    - The paper specifies a *squared* L2 penalty, ||m||_2^2. The previous
      implementation used the un-squared L2 norm ||m||_2 which (a) does not
      match the paper and (b) has a different gradient scale: for a learnable
      mask token of dimension D with elements ~ N(2, 1) and D=1000, the
      un-squared norm is ~70 with a gradient magnitude of ~m_i/||m|| ~= 0.03
      per element. The squared norm is more standard for parameter
      regularization and provides a 2*m_i gradient per element.
    """
    # Squared L2 norm matches the paper. `mask_token` is typically 1D
    # (shape [num_features]); summing the squares is equivalent to the
    # squared 2-norm regardless of dimensionality.
    mask_norm_sq = mask_token.pow(2).sum()
    return wt_mask_token_regularization * mask_norm_sq