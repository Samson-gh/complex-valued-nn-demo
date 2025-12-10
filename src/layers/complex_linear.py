import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """Complex-valued linear layer scaffold."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real_feat = torch.nn.Linear(in_features, out_features, bias=bias)
        self.imag_feat = torch.nn.Linear(in_features, out_features, bias=bias)
        
def forward(self, x):
    # return complex_matmul(x, self.real_feat.weight, self.imag_feat.weight)
    # TODO: Implement complex matmul and bias addition

    raise NotImplementedError("Forward method not implemented yet.")