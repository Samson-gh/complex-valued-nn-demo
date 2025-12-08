import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """Complex-valued linear layer scaffold."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # TODO: Implement real/imag parameter initialization


def forward(self, x):
# TODO: Implement complex matmul
    raise NotImplementedError