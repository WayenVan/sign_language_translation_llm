import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class FullFreezer:
    def __init__(self, visual_encoder: nn.Module):
        self.visual_encoder = visual_encoder

    def freeze(self):
        """
        Freeze the visual encoder.
        """
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()

    def train(self, is_training):
        self.visual_encoder.eval()


class NoFreezer:
    def __init__(self, visual_encoder: nn.Module):
        """
        do not need to store anything
        """

    def freeze(self):
        """
        do not freeze naything
        """
        pass

    def train(self, is_training):
        pass


class LoraFreezer:
    def __init__(self, visual_encoder: nn.Module, freeze_adapter: bool = False):
        """
        do not need to store anything
        """
        self.freeze_adapter = freeze_adapter
        self.visual_encoder = visual_encoder

    def freeze(self):
        """
        do not freeze naything
        """
        if self.freeze_adapter:
            logger.info("Freezing LoRA parameters in visual encoder")
            for name, param in self.visual_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = False
            assert self.visual_encoder.get_nb_trainable_parameters()[0] == 0, (
                "There are still trainable parameters in the visual encoder, please check the implementation."
            )
        else:
            logger.info("Not freezing LoRA parameters in visual encoder")

    def train(self, is_training):
        pass
