import sys
import os
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # NOTE: this is the initial cwd when runing the sciprt

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

import datetime

from misc.git_utils import save_git_info
from typing import Any, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))
logger = logging.getLogger(__name__)


# NOTE: the hydra appp only inisitalize once
@hydra.main(config_path="../configs", config_name="initial_train", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    train(cfg, hydra_config)


def init_output_dir(config_name: str) -> str:
    """
    Initialize the output directory for the job.
    """
    now = datetime.datetime.now()
    subfolder = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", config_name, subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def init_logger(local_rank, output_dir: str) -> WandbLogger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Stream handler for console output
            logging.FileHandler(
                os.path.join(output_dir, f"train_rank{local_rank}.log")
            ),  # File handler for logging to a file
        ],
    )
    return logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    hydra_config: DictConfig,
) -> None:
    config_name = hydra_config.job.config_name

    # NOTE: set the output directory
    output_dir = os.environ.get(
        config_name.upper() + "_OUTPUT_DIR",
        None,
    )
    if output_dir is None:
        print(f"Output directory not found in environment variables, initializing...")
        output_dir = init_output_dir(config_name)
        os.environ[config_name.upper() + "_OUTPUT_DIR"] = output_dir

    init_logger(local_rank, output_dir)
    logger.info(f"Output directory: {output_dir}")

    # NOTE: set the logger
    wdb_config = OmegaConf.to_container(cfg, resolve=True)
    wdb_config["output_dir"] = output_dir
    lt_logger = WandbLogger(
        name=config_name,
        project="sign-langauge-translation-llm",
        config=wdb_config,
    )
    run_id = str(lt_logger.experiment.id)

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        callbacks.LearningRateMonitor("step", log_momentum=True),
        callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename=run_id + "-{epoch:02d}-{val_generate_bleu:.4f}",
            monitor="val_generate_bleu",
            mode="max",
            save_last=True,
            save_weights_only=True,
        ),
        # DebugCallback(),
    ]

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=getattr(cfg, "devices", "auto"),
        callbacks=cbs,
        log_every_n_steps=50,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,  # NOTE: gradient clipping will be normed
        # gradient_clip_algorithm="value",
        sync_batchnorm=True,
        precision=cfg.precision,
        logger=lt_logger,
        # WARN: will slow down the training process, just for debug now
        # detect_anomaly=True,
    )

    if t.is_global_zero:
        # NOTE: save git info
        save_git_info(
            repo_path=project_root,
            info_path=os.path.join(output_dir, "git_info"),
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
            # trainer.should_stop = True

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Any,
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
