import torch
import torch.nn as nn
from src.layers.apply_complex import apply_complex


class ComplexLinear(nn.Module):
    """Complex-valued linear layer scaffold."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real_feat = torch.nn.Linear(in_features, out_features, bias=bias)
        self.imag_feat = torch.nn.Linear(in_features, out_features, bias=bias)
        
def forward(self, x):
    apply_complex(x, self.real_feat.weight, self.imag_feat.weight)
    # TODO: Implement complex matmul and bias addition

    raise NotImplementedError("Forward method not implemented yet.")