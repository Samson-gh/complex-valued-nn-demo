import torch

def apply_complex(input, real_weight, imag_weight, dtype=torch.complex64):
    """Performs complex matrix multiplication.

    Args:
        input (torch.Tensor): Input tensor of shape (..., in_features) with complex dtype.
        real_weight (torch.Tensor): Real part of weight matrix of shape (out_features, in_features).
        imag_weight (torch.Tensor): Imaginary part of weight matrix of shape (out_features, in_features).
        dtype (torch.dtype): Complex data type for computation.

    Returns:
        torch.Tensor: Resulting tensor after complex matrix multiplication.
    """
    return (real_weight(input.real)-imag_weight(input.imag)).type(dtype) \
        + 1j*(real_weight(input.imag)+imag_weight(input.real)).type(dtype)