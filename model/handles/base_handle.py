from lightning.pytorch import LightningModule
from torch import nn


class BaseHandle(nn.Module):
    def train_step(self, module: LightningModule, batch, batch_idx):
        pass

    def validation_step(self, module: LightningModule, batch, batch_idx):
        pass

    def test_step(self, module: LightningModule, batch, batch_idx):
        pass
