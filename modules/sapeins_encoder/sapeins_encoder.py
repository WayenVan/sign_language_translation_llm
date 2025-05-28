from mmpose.apis import init_model
import torchvision.transforms.functional as F
import torch
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmengine.config import Config
from einops import rearrange
import numpy as np
from collections import namedtuple


class SapeinsVisualEncoder(torch.nn.Module):
    def __init__(self, cfg, ckpt, hidden_states_layer):
        super().__init__()
        self.id = id
        self.cfg = Config.fromfile(cfg)
        self.ckpt = ckpt
        self.hidden_states_layer = hidden_states_layer

        self.model: VisionTransformer = init_model(cfg, ckpt, device="cpu").backbone

        # NOTE: setup the output indices
        self.model.out_indices = tuple(range(len(self.model.layers)))
        self.model.out_type = "raw"

        # NOTE: normalization

    @torch.no_grad()
    def _normalize(self, video):
        # video: (B, T, C, H, W)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32) / 255.0
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32) / 255.0
        video = F.normalize(
            video,
            mean=mean.tolist(),
            std=std.tolist(),
        )
        return video

    SapeinsVisualEncoderOutput = namedtuple(
        "ViTPoseVisualEncoderOutput", ["hidden_state", "video_length"]
    )

    def forward(
        self,
        video,
        video_length,
    ):
        # video: (B, T, C, H, W)
        B, T, C, H, W = video.shape

        video = rearrange(video, "b t c h w -> (b t) c h w")
        normalized_video = self._normalize(video)
        outputs = self.model(
            normalized_video,
        )
        return self.SapeinsVisualEncoderOutput(
            hidden_state=rearrange(
                outputs[self.hidden_states_layer],
                "(b t) hw d -> b t hw d",
                b=B,
                t=T,
            ),
            video_length=video_length,
        )


if __name__ == "__main__":
    from PIL import Image

    model = SapeinsVisualEncoder(
        cfg="sapeins_configs/sapiens_pose/coco_wholebody/sapiens_0.3b-210e_coco_wholebody-1024x768.py",
        ckpt="outputs/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth",
        hidden_states_layer=-1,  # last layer
    )

    model.cuda()
    video = Image.open("outputs/visualization_val/110.jpg")
    video = video.convert("RGB")
    video = F.resize(video, (512, 384), antialias=True)
    video = F.to_tensor(video)
    video = video.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    video = video.repeat(1, 10, 1, 1, 1)  # (1, 10, C, H, W)
    video_length = torch.tensor([10], dtype=torch.int64)  # (B,)

    output = model(video.cuda(), video_length.cuda())
    print("Hidden state shape:", output.hidden_state.shape)  # (B, T, HW, D)
