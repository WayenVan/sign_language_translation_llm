from collections import namedtuple
import torch
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
import logging

import re

from transformers import (
    AutoConfig,
)

if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from modules.vitpose_visual_lora_encoder.modeling_vitpose import (
        # VitPoseForPoseEstimation,
        VitPoseForFeatureExtraction,
    )
else:
    from .modeling_vitpose import (
        VitPoseForPoseEstimation,
        VitPoseForFeatureExtraction,
    )

logger = logging.getLogger(__name__)


class VitPoseLoraVisualEncoder(torch.nn.Module):
    def __init__(
        self,
        id,
        hidden_states_layer,
        lora_rank,
        lora_alpha,
        start_lora_layer=0,
        lora_dropout=0.1,
    ):
        super().__init__()
        self.id = id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.start_lora_layer = start_lora_layer
        self._init_lora_model()

        self.vitpose_cfg = AutoConfig.from_pretrained(id)
        self.image_size = self.vitpose_cfg.backbone_config.image_size
        self.hidden_size = self.vitpose_cfg.backbone_config.hidden_size
        self.num_keypoints = self.vitpose_cfg.num_labels
        self.hidden_states_layer = hidden_states_layer

    ViTPoseVisualEncoderOutput = namedtuple(
        "ViTPoseVisualEncoderOutput", ["hidden_state", "video_length", "heatmaps"]
    )

    def _init_lora_model(self):
        model: VitPoseForFeatureExtraction = (
            VitPoseForFeatureExtraction.from_pretrained(self.id)
        )
        target_modules = []
        # backbone.encoder.layer.1.attention.attention.query Linear
        for name, module in model.named_modules():
            match = re.match(r"backbone\.encoder\.layer\.([0-9]+)", name)
            if match and int(match.group(1)) >= self.start_lora_layer:
                if isinstance(module, torch.nn.Linear):
                    target_modules.append(name)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
        )

        self.model = get_peft_model(
            model,
            lora_config,
        )
        trainable, all = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Created Lora VitPose for {self.id} Trainable parameters: {trainable}, All parameters: {all}, Ratio: {trainable / all:.2%}"
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
            pixel_values=video,
            dataset_index=torch.zeros(B * T).long().to(video.device),  # 0 is the best
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
            heatmaps=None,
            # heatmaps=rearrange(outputs.heatmaps, "(b t) k h w -> b t k h w", b=B, t=T),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = VitPoseLoraVisualEncoder(
        id="usyd-community/vitpose-plus-base",
        hidden_states_layer=-2,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        start_lora_layer=2,
    ).cuda(0)
    for name, param in model.model.named_parameters():
        print(name, param.requires_grad)

    # video = torch.randn(2, 10, 3, 256, 192).cuda(0)  # (B, T, C, H, W)
    # video_length = torch.tensor([10, 8]).cuda(0)  # Example video lengths for each batch
    # output = model(video, video_length)
    # print("Output hidden state shape:", output.hidden_state.shape)
