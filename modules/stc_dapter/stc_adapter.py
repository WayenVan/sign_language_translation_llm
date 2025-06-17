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
        feature_size: tuple[int, int] = (14, 14),
        downsample: tuple[int, int, int] = (2, 2, 2),
        max_length=512,
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
        self.proj = MLP(
            hidden_size,
            [hidden_size, target_hidden_size],
            norm_layer=nn.LayerNorm,
            activation_layer=nn.GELU,
        )
        self.H = feature_size[0]
        self.W = feature_size[1]

    def forward(self, x, v_length):
        # x: (B, T, HW, C)
        B, T, HW, C = x.shape
        assert HW == self.H * self.W, "HW must match H * W"
        x = rearrange(x, "b t (h w) c -> (b t) c h w", h=self.H, w=self.W)
        x = self.s1(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)
        x = self.downsample(x)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.s2(x)
        x = rearrange(x, "(b t) c h w -> b t (h w) c", b=B)
        x = reduce(x, "b t hw c -> b t c", "mean")
        x = self.proj(x)
        v_lenth = v_length // self._downsample[0]
        return x, v_lenth


if __name__ == "__main__":
    # Example usage
    device = "cuda:6"
    B, T, HW, C = 2, 128, 196, 512  # Batch size, video length, feature dimension
    hidden_size = 512
    target_hidden_size = 256
    depth = 2
    feature_size = (14, 14)
    downsample = (2, 2, 2)
    adapter = SpatialTemporalAdapter(
        hidden_size, target_hidden_size, depth, feature_size, downsample
    ).to(device)
    x = torch.randn(B, T, HW, C).to(device)
    v_length = torch.tensor([10, 8]).to(device)  # Video lengths
    output = adapter(x, v_length)
