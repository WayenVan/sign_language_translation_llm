from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)
from torch import nn
import torch

from einops import rearrange


class DinoV2Backbone(nn.Module):
    def __init__(self, id, output_layer=-1):
        super().__init__()
        self.visual_encoder = Dinov2WithRegistersModel.from_pretrained(id)
        self.output_layer = output_layer

    def forward(self, video):
        """
        videoo: [B, T, C, H, W]
        """
        B, T, C, H, W = video.shape
        video = rearrange(video, "b t c h w -> (b t) c h w")
        feats = self.visual_encoder(video, output_hidden_states=True).hidden_states[
            self.output_layer
        ]
        feats = rearrange(feats, "(b t) hw c -> b t hw c", b=B, t=T)

        return feats


if __name__ == "__main__":
    model = DinoV2Backbone("facebook/dinov2-with-registers-base")
    print(model)
    video = torch.randn(2, 8, 3, 224, 224)  # Example video tensor
    feats = model(video)
    print(feats.shape)
