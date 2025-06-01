import torch
from torch import nn
from einops import rearrange, reduce, repeat
import timm
from collections import namedtuple


class TimmVisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_id,
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

    TimmVisualEncoderOutput = namedtuple(
        "TimmVisualEncoderOutput", ["hidden_state", "video_length"]
    )

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
        return self.TimmVisualEncoderOutput(
            hidden_state=visual_features,
            video_length=v_length,
        )


if __name__ == "__main__":
    # Example usage
    model = TimmVisualEncoder("resnet18", out_channels=512)
    video_tensor = torch.randn(
        2, 10, 3, 224, 224
    )  # [batch_size, time, channels, height, width]
    v_length = torch.tensor([10, 10])  # Length of each video in the batch
    features, lengths = model(video_tensor, v_length)
    print(features.shape)  # Should print: torch.Size([2, 10, 512])
    print(lengths)  # Should print: tensor([10, 10])
