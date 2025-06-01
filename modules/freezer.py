import torch
from torch import nn


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

