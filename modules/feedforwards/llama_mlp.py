from transformers.activations import ACT2FN
from torch import nn


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn, mlp_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=mlp_bias
        )
        self.act_fn = ACT2FN[act_fn]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
