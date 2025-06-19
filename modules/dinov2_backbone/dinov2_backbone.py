from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)
from torch import nn
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig
import re

from einops import rearrange

from typing import NamedTuple, List

import logging

logger = logging.getLogger(__name__)


class DinoV2Backbone(nn.Module):
    def __init__(self, id, output_layer=-1, start_lora_layer=None, **lora_kwargs):
        super().__init__()
        self.id = id
        if start_lora_layer is None:
            self.visual_encoder = Dinov2WithRegistersModel.from_pretrained(id)
        else:
            self.start_lora_layer = start_lora_layer
            self._init_lora_model(lora_kwargs)

        self.output_layer = output_layer

    def _init_lora_model(self, lora_kwargs):
        visual_encoder = Dinov2WithRegistersModel.from_pretrained(self.id)
        target_modules = []
        # backbone.encoder.layer.1.attention.attention.query Linear
        for name, module in visual_encoder.named_modules():
            match = re.match(r"encoder\.layer\.([0-9]+)", name)
            if match and int(match.group(1)) >= self.start_lora_layer:
                if isinstance(module, torch.nn.Linear):
                    target_modules.append(name)

        lora_config = LoraConfig(
            bias="none",
            # task_type=TaskType.IMAGE_CLASSIFICATION,
            target_modules=target_modules,
            # lora_alpha=self.lora_alpha,
            # lora_dropout=self.lora_dropout,
            # r=self.lora_rank,
            **lora_kwargs,
        )

        self.visual_encoder = get_peft_model(
            visual_encoder,
            lora_config,
        )
        trainable, all = self.visual_encoder.get_nb_trainable_parameters()
        logger.info(
            f"Created Lora dino for {self.id} Trainable parameters: {trainable}, All parameters: {all}, Ratio: {trainable / all:.2%}"
        )

    def forward(self, video, v_length):
        """
        videoo: [B, T, C, H, W]
        """
        B, T, C, H, W = video.shape
        video = rearrange(video, "b t c h w -> (b t) c h w")
        feats = self.visual_encoder(video, output_hidden_states=True).hidden_states[
            self.output_layer
        ]
        feats = rearrange(feats, "(b t) hw c -> b t hw c", b=B, t=T)

        return feats, v_length


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = DinoV2Backbone(
        "facebook/dinov2-with-registers-base",
        start_lora_layer=6,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    # print(model)
    video = torch.randn(2, 8, 3, 224, 224)  # Example video tensor
    feats = model(video)
    print(feats.shape)
