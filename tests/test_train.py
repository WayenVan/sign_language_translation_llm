import sys
import os
import logging
import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # NOTE: this is the initial cwd when runing the sciprt, the hydra will change the cwd to the output dir

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.optim import Optimizer


from lightning import Trainer
from lightning.pytorch import callbacks
import lightning.pytorch as pl

from model.slt import SLTModel
import cv2
from typing import Any


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error


# NOTE: the hydra appp only inisitalize once
@hydra.main(
    config_path="../configs", config_name="prompt_learning", version_base="1.3.2"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


def train(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    working_dir = hydra_config.runtime.output_dir
    config_name = hydra_config.job.config_name

    logger.info(f"Output directory: {working_dir}")

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        DebugCallback(),
    ]

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=[0],
        callbacks=cbs,
        log_every_n_steps=50,
        # max_steps=
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,  # NOTE: gradient clipping will be normed
        # gradient_clip_algorithm="value",
        sync_batchnorm=True,
        precision="16-mixed",
        logger=None,
        # detect_anomaly=True,
    )

    logger.info(f"Process in local rank {t.local_rank}, global rank {t.global_rank}")

    datamodule = instantiate(cfg.data.datamodule, cfg)
    model = SLTModel(cfg)
    t.fit(model, datamodule=datamodule)


class DebugCallback(callbacks.Callback):
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        # vdieo = batch["video"]
        # logging.info(
        #     f"Video shape: {vdieo.shape}, mean: {vdieo.mean()}, std: {vdieo.std()}"
        # )
        # val_steps = 0
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
    ) -> None:
        # NOTE: check the gradient norm
        #
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            logging.info(f"Param {name} has  mean: {param.mean()}, std: {param.std()}")

            if param.grad is not None:
                logging.info(
                    f"Param {name} has grad mean: {param.grad.mean()}, std: {param.grad.std()}"
                )
            else:
                logging.info(f"Param {name} has no grad")
        if trainer.global_step > 100:
            trainer.should_stop = True

        # NOTE: check nan
        for name, param in pl_module.named_parameters():
            global_step = trainer.global_step

            if torch.isnan(param).any():
                logger.warning(
                    f"In Step {global_step}, Param {name} has mean: {param.mean()}, std: {param.std()}"
                )

            if param.grad is not None and torch.isnan(param.grad).any():
                logger.warning(
                    f"In Step {global_step}, Param {name} has grad mean: {param.grad.mean()}, std: {param.grad.std()}"
                )
        return


if __name__ == "__main__":
    main()
