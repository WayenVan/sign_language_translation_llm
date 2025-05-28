import torch
from torch import nn
from einops import rearrange, reduce, repeat
import timm
from .tconv import TemporalConv1D


class TimmVisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_id,
        out_channels,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_id,
            pretrained=True,
            num_classes=0,
            **kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.backbone_out_feautres = self.backbone.num_features

    def forward(self, x, v_length=None):
        """
        @param x: the input video tensor [batch_size, time, 3, height, width]
        @param v_length: the length of the video sequence [batch_size]
        """
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w").contiguous()
        visual_features = self.backbone(x)
        visual_features = self.dropout(visual_features)
        visual_features = rearrange(
            visual_features, "(b t) c -> b t c", b=B, t=T
        ).contiguous()
        return visual_features, v_length
