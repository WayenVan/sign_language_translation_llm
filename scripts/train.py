import sys
import os
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # NOTE: this is the initial cwd when runing the sciprt, the hydra will change the cwd to the output dir

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lightning import Trainer
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch

from model.slt import SLTModel
import cv2

from misc.git_utils import save_git_info
from typing import Any, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error


# NOTE: the hydra appp only inisitalize once
@hydra.main(config_path="../configs", config_name="initial_train", version_base="1.3.2")
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
        callbacks.LearningRateMonitor("step", log_momentum=True),
        callbacks.ModelCheckpoint(
            dirpath=working_dir,
            filename="epoch={epoch:02d}-wer={val_token_level_accu:.2f}",
            monitor="val_generate_accu",
            mode="max",
            save_last=True,
            save_weights_only=True,
        ),
        DebugCallback(),
    ]

    # NOTE: set the logger
    wdb_config = OmegaConf.to_container(cfg, resolve=True)
    wdb_config["output_dir"] = working_dir
    lt_logger = WandbLogger(
        name=config_name,
        project="sign-langauge-translation-llm",
        config=wdb_config,
    )

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=getattr(cfg, "devices", "auto"),
        callbacks=cbs,
        log_every_n_steps=50,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,  # NOTE: gradient clipping will be normed
        gradient_clip_algorithm="value",
        sync_batchnorm=True,
        precision="16-mixed",
        logger=lt_logger,
        # WARN: will slow down the training process, just for debug now
        # detect_anomaly=True,
    )

    if t.is_global_zero:
        # NOTE: save git info
        save_git_info(
            repo_path=project_root,
            info_path=os.path.join(working_dir, "git_info"),
        )

    logger.info(f"Process in local rank {t.local_rank}, global rank {t.global_rank}")

    datamodule = instantiate(cfg.data.datamodule, cfg)
    model = SLTModel(cfg)
    t.fit(model, datamodule=datamodule)


class DebugCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.current_train_batch = None

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.current_train_batch = batch

    def on_before_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor
    ) -> None:
        # NOTE: check the loss
        if torch.isnan(loss).any():
            video = self.current_train_batch["video"]
            ids = self.current_train_batch["ids"]

            logger.warning(f"Loss is NaN: {loss}")
            logger.warning(
                f"Video shape: {video.shape}, mean: {video.mean()}, std: {video.std()}"
            )
            logger.warning(f"input_ids: {ids}")
            trainer.should_stop = True

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: any,
    ) -> None:
        nan_flag = False
        for name, param in pl_module.named_parameters():
            global_step = trainer.global_step

            if torch.isnan(param).any():
                nan_flag = True
                logger.warning(
                    f"In Step {global_step}, Param {name} has mean: {param.mean()}, std: {param.std()}"
                )
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_flag = True
                logger.warning(
                    f"In Step {global_step}, Param {name} has grad mean: {param.grad.mean()}, std: {param.grad.std()}"
                )
        # if nan_flag and global_step >= 1000:
        #     logger.warning(
        #         "find nan and the global step is larger than 1000, stop the training"
        #     )
        #     trainer.should_stop = True
        return


if __name__ == "__main__":
    main()
