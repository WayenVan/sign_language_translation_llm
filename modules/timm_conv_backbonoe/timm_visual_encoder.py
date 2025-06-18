import torch
from torch import nn
from einops import rearrange, reduce, repeat
import timm


class TimmConvBackbone(nn.Module):
    def __init__(
        self,
        id,
        **kwargs,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            id,
            pretrained=True,
            num_classes=0,
            in_chans=3,
            **kwargs,
        )
        self.backbone_out_feautres = self.backbone.num_features
        # self.out_channels = out_channels
        # self.backbone.head = nn.Linear(self.backbone.head.in_features, out_channels)

    def forward(self, x):
        """
        @param x: the input video tensor [batch_size, time, 3, height, width]
        @param v_length: the length of the video sequence [batch_size]
        """
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w").contiguous()
        visual_features = self.backbone(x)
        visual_features = rearrange(visual_features, "(b t) c -> b t c", b=B, t=T)
        return visual_features


if __name__ == "__main__":
    model = TimmConvBackbone("resnet50")
    video = torch.randn(
        2, 10, 3, 224, 224
    )  # [batch_size, time, channel, height, width]
    features = model(video)
    print(features.shape)  # Should print [batch_size * time, out_channels]
