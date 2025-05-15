from collections import namedtuple
import torch
from einops import rearrange


from transformers import (
    AutoConfig,
    VitPoseForPoseEstimation,
)


class VitPoseVisualEncoder(torch.nn.Module):
    def __init__(self, id, hidden_states_layer):
        super(VitPoseVisualEncoder, self).__init__()
        self.id = id
        self.model: VitPoseForPoseEstimation = VitPoseForPoseEstimation.from_pretrained(
            id
        )
        self.vitpose_cfg = AutoConfig.from_pretrained(id)
        self.image_size = self.vitpose_cfg.backbone_config.image_size
        self.hidden_size = self.vitpose_cfg.backbone_config.hidden_size
        self.num_keypoints = self.vitpose_cfg.num_labels
        self.hidden_states_layer = hidden_states_layer

    ViTPoseVisualEncoderOutput = namedtuple(
        "ViTPoseVisualEncoderOutput", ["hidden_states", "video_length", "heatmaps"]
    )

    def forward(
        self,
        video,
        video_length,
    ):
        # video: (B, T, C, H, W)
        B, T, C, H, W = video.shape
        assert (H, W) == (self.image_size[0], self.image_size[1]), (
            f"Input video size {H}x{W} does not match model size {self.image_size[0]}x{self.image_size[1]}"
        )
        video = rearrange(video, "b t c h w -> (b t) c h w")
        outputs = self.model(
            video,
            dataset_index=torch.zeros(B * T).long(),  # 0 is the best
            output_hidden_states=True,
        )
        return self.ViTPoseVisualEncoderOutput(
            hidden_state=rearrange(
                outputs.hidden_states[self.hidden_states_layer],
                "(b t) hw d -> b t hw d",
                b=B,
                t=T,
            ),
            video_length=video_length,
            heatmaps=rearrange(outputs.heatmaps, "(b t) k h w -> b t k h w", b=B, t=T),
        )
