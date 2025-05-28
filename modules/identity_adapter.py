import torch
from torch import nn


class IdentityAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, v_length=None):
        """
        @param x: the input tensor
        @param v_length: the length of the video sequence (not used)
        """
        # Simply return the input tensor as is
        return x, v_length
