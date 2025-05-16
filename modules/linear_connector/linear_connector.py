import torch
from torch import nn


class LinearConnector(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.proj(x)
