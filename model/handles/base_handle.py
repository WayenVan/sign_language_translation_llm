from lightning.pytorch import LightningModule
from torch import nn


class BaseHandle(nn.Module):
    def train_step(self, module: LightningModule, batch, batch_idx):
        pass

    def validation_step(self, module: LightningModule, batch, batch_idx):
        pass

    def test_step(self, module: LightningModule, batch, batch_idx):
        pass

    def train_handle(self, module: LightningModule, is_train: bool):
        pass

    def on_train_epoch_end(self, module: LightningModule):
        pass

    def on_validation_epoch_end(self, module: LightningModule):
        pass
