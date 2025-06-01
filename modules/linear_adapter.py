import torch
from torch import nn
from torch.nn import functional as F


class LinearAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, v_length=None):
        """
        @param x: [b t c]
        @param v_length: the length of the video sequence (not used)
        """
        # Simply return the input tensor as is
        x = F.normalize(x, dim=-1, p=2)
        x = self.linear(x)
        return x, v_length
