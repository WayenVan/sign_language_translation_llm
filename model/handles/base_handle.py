from lightning.pytorch import LightningModule
from torch import nn


class BaseHandle(nn.Module):
    def train_step(self, module: LightningModule, batch, batch_idx):
        """
        Handles the training step for the MLM task.
        """

    def validation_step(self, module: LightningModule, batch, batch_idx):
        """
        Handles the validation step for the MLM task.
        """

    def test_step(self, module: LightningModule, batch, batch_idx):
        """
        Handles the test step for the MLM task.
        """
