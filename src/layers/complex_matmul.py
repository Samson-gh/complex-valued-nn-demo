import torch

def complex_matmul(input, real_weight, imag_weight, dtype=torch.complex64):
    """Performs complex matrix multiplication.

    Args:
        input (torch.Tensor): Input tensor of shape (..., in_features) with complex dtype.
        real_weight (torch.Tensor): Real part of weight matrix of shape (out_features, in_features).
        imag_weight (torch.Tensor): Imaginary part of weight matrix of shape (out_features, in_features).
        dtype (torch.dtype): Complex data type for computation.

    Returns:
        torch.Tensor: Resulting tensor after complex matrix multiplication.
    """
    # Convert weights to complex dtype
    weight = torch.complex(real_weight, imag_weight).to(dtype)

    # Perform complex matrix multiplication
    output = torch.matmul(input.to(dtype), weight.T)

    return output