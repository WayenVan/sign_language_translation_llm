import torch
from einops import rearrange, repeat, reduce
from torch import nn
from torchvision.ops import MLP

from typing import Optional, Type
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert import BertConfig

from timm.models.regnet import RegStage


class SpatialTemporalAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        depth,
        mlp_depth,
        feature_size: tuple[int, int] = (14, 14),
        downsample: tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self._downsample = downsample

        self.s1 = RegStage(
            depth=depth,
            in_chs=hidden_size,
            out_chs=hidden_size,
            stride=1,
            dilation=1,
        )
        self.downsample = nn.Conv3d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=downsample,
            stride=downsample,
        )
        self.s2 = RegStage(
            depth=depth,
            in_chs=hidden_size,
            out_chs=hidden_size,
            stride=1,
            dilation=1,
        )
        self.proj = build_mlp(mlp_depth, hidden_size, target_hidden_size)
        self.H = feature_size[0]
        self.W = feature_size[1]
        self.pool = ScoredPooling(hidden_size)

    def forward(self, x, v_length):
        # x: (B, T, HW, C)
        B, T, HW, C = x.shape
        assert HW == self.H * self.W, "HW must match H * W"
        x = rearrange(x, "b t (h w) c -> (b t) c h w", h=self.H, w=self.W).contiguous()
        x = self.s1(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T).contiguous()
        x = self.downsample(x)
        x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        x = self.s2(x)
        x = rearrange(x, "(b t) c h w -> b t h w c ", b=B).contiguous()
        x = self.pool(x)  # Apply pooling to reduce spatial dimensions
        x = self.proj(x)
        v_lenth = v_length // self._downsample[0]
        return x, v_lenth


class ScoredPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (..., H, W, C)
        score = self.score(x)  # (..., H, W, 1)
        return (x * score).mean(dim=(-3, -2))


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


if __name__ == "__main__":
    # Example usage
    device = "cuda:0"
    B, T, HW, C = 2, 128, 196, 512  # Batch size, video length, feature dimension
    hidden_size = 512
    target_hidden_size = 256
    depth = 2
    feature_size = (14, 14)
    downsample = (2, 2, 2)
    adapter = SpatialTemporalAdapter(
        hidden_size, target_hidden_size, depth, 2, feature_size, downsample
    ).to(device)
    x = torch.randn(B, T, HW, C).to(device)
    v_length = torch.tensor([128, 120]).to(device)  # Example video lengths
    output = adapter(x, v_length)
