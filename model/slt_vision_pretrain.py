from einops import rearrange
import torch
from torch import nn, Tensor
from lightning import LightningModule

from omegaconf import OmegaConf, DictConfig

from hydra.utils import instantiate

from typing import List
from transformers import get_scheduler
from torch.optim import Optimizer
from torch.nn import functional as F


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.BatchNorm1d(output_hidden_size))
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class SignBackboneForVPretraining(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg: DictConfig = cfg
        self.visual_backbone = instantiate(self.cfg.modules.visual_encoder)
        self.visual_adapter = instantiate(self.cfg.modules.visual_adapter)

        self.shifts = cfg.shifts
        self.visual_hidden_size = self.cfg.modules.visual_hidden_size

        self.forward_predictor = build_mlp(
            self.cfg.modules.predictor_depth,
            self.visual_hidden_size,
            self.visual_hidden_size,
        )
        self.backward_predictor = build_mlp(
            self.cfg.modules.predictor_depth,
            self.visual_hidden_size,
            self.visual_hidden_size,
        )

        # for param in self.visual_backbone.parameters():
        #     param.requires_grad = False
        # self.visual_backbone.eval()

    def dispatch_batch(self, batch, device):
        ids: list[str] = batch["id"]
        video: Tensor = batch["video"].to(device)
        video_aug1 = batch["video_aug1"].to(device)
        video_aug2 = batch["video_aug2"].to(device)
        video_length: Tensor = batch["video_length"].to(device)
        text: list[str] = batch["text"]
        return ids, video_aug1, video_aug2, video_length, text

    def forward(
        self,
        video: Tensor,
        video_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the visual encoder and adapter.
        """
        # with torch.no_grad():
        feats = self.visual_backbone(video)
        # feats = self.visual_adapter(feats)
        return feats

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        ids, video_aug1, video_aug2, video_length, text = self.dispatch_batch(
            batch, self.device
        )
        feats1 = self.forward(video_aug1, video_length)
        feats2 = self.forward(video_aug2, video_length)

        # Calculate loss for each shift
        loss = self.calculate_loss(self.shifts, feats1, feats2, video_length)
        self.log("train_siam_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        ids, video_aug1, video_aug2, video_length, text = self.dispatch_batch(
            batch, self.device
        )
        # aug1 == aug2, so we can use either one
        feats = self.forward(video_aug1, video_length)  # [B, T, C]

        # Calculate loss for each shift
        loss = self.calculate_loss(self.shifts, feats, feats, video_length)

        self.log("val_siam_loss", loss, on_epoch=True, prog_bar=True)

        stds = rearrange(feats, "b t c -> (b t) c").std(dim=1).mean()
        self.log(
            "val_feats_std",
            stds,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    @staticmethod
    def get_lr_schuduler(cfg, optimizer: Optimizer):
        lr_config = cfg.engine.lr_scheduler
        if lr_config.type == "native" or lr_config.type == "timm":
            return instantiate(lr_config.instance, optimizer=optimizer)
        elif lr_config.type == "transformers":
            # scheduler = instantiate(self.cfg.engine.lr_scheduler, opt)
            kwarges = cfg.engine.lr_scheduler.kwargs
            if kwarges is None:
                kwarges = {}
            else:
                kwarges = OmegaConf.to_container(kwarges, resolve=True)

            scheduler = get_scheduler(
                cfg.engine.lr_scheduler.name,
                optimizer=optimizer,
                num_warmup_steps=cfg.engine.lr_scheduler.warmup_steps,
                num_training_steps=cfg.engine.lr_scheduler.training_steps,
                scheduler_specific_kwargs=kwarges,
            )
            return scheduler
        else:
            raise ValueError(
                f"Unsupported lr scheduler type: {cfg.engine.lr_scheduler.type}"
            )

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.engine.optimizer,
            [
                {"params": filter(lambda p: p.requires_grad, self.parameters())},
            ],
        )
        scheduler = self.get_lr_schuduler(self.cfg, opt)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'epoch'
            },
        }

    @staticmethod
    def length_to_mask(lengths, max_length=None):
        """
        Convert lengths to a boolean mask.
        lengths: [B]
        max_length: int, optional
        """
        if max_length is None:
            max_length = lengths.max().item()
        B = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(
            B, max_length
        ) < lengths.unsqueeze(1)
        return mask.long()  # (B, max_length)

    def calculate_loss(self, shifts, feats1, feats2, v_length):
        """
        feats: [B, T, C]
        """
        B, T, C = feats1.size()
        forward_shift = feats1[:, shifts:]
        backward_shift = feats2[:, :-shifts]

        feats_mask = self.length_to_mask(v_length, max_length=feats1.size(1))
        forward_mask = feats_mask[:, shifts:]
        backward_mask = feats_mask[:, :-shifts]

        mask = forward_mask * backward_mask

        _forward_shift = rearrange(
            forward_shift,
            "b t c -> (b t) c",
        )
        _backward_shift = rearrange(
            backward_shift,
            "b t c -> (b t) c",
        )
        forward_pred = self.forward_predictor(_forward_shift)
        backward_pred = self.backward_predictor(_backward_shift)

        forward_pred = rearrange(
            forward_pred,
            "(b t) c -> b t c",
            b=B,
            t=T - shifts,
        )
        backward_pred = rearrange(
            backward_pred,
            "(b t) c -> b t c",
            b=B,
            t=T - shifts,
        )

        forward_loss = (
            self.distance(forward_pred, backward_shift, mask) / 2
            + self.distance(backward_pred, forward_shift, mask) / 2
        )
        return forward_loss

    @staticmethod
    def distance(predicted: Tensor, target: Tensor, padding_mask) -> Tensor:
        """
        Calculate the distance between predicted and target features.
        shape: [B, T, C]
        padding_mask: [B, T], where 1 means valid and 0 means padding
        """
        target = target.detach()

        predicted = F.normalize(predicted, dim=-1)  # [B, T, C]
        target = F.normalize(target, dim=-1)

        # sim = -torch.einsum("btc,btc->bt", predicted, target)  # [B, T]
        sim = -F.cosine_similarity(predicted, target, dim=-1)  # [B, T]

        assert (padding_mask.sum(dim=-1) > 0).all(), (
            "Padding mask should not be all zeros"
        )
        sim = sim * padding_mask.float()  # [B, T]
        sim = sim.sum(dim=-1) / padding_mask.sum(dim=-1)  # [B]
        return sim.mean()
