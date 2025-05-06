import torch
from torch import nn
from einops import rearrange, reduce, repeat
import timm
from .tconv import TemporalConv1D


class TimmVisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_id,
        img_size,
        out_channels,
        dropout_s=0.5,
        drop_path_rate_s=0.1,
        dropout_t=0.5,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_id,
            pretrained=True,
            num_classes=0,
            in_chans=3,
            img_size=img_size,
            drop_rate=dropout_s,
            drop_path_rate=drop_path_rate_s,
        )
        self.backbone_out_feautres = self.backbone.num_features
        self.tconv = TemporalConv1D(
            input_size=self.backbone_out_feautres,
            out_size=out_channels,
            bottleneck_size=out_channels,
            conv_type=["K3", "P2", "K3", "P2"],
            pooling="max",
            dropout=dropout_t,
        )
        # self.out_channels = out_channels
        # self.backbone.head = nn.Linear(self.backbone.head.in_features, out_channels)

    def forward(self, x, v_length=None):
        """
        @param x: the input video tensor [batch_size, time, 3, height, width]
        @param v_length: the length of the video sequence [batch_size]
        """
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w").contiguous()
        visual_features = self.backbone(x)
        visual_features = rearrange(visual_features, "(b t) c -> b c t", b=B)
        visual_features, v_length = self.tconv(visual_features, v_length)
        visual_features = rearrange(visual_features, "b c t -> b t c")
        return visual_features, v_length
