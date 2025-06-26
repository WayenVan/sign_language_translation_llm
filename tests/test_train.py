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

from model import SLTModelForT5FineTune, ModelForT5TextPretrain
from model.mbart_slt import MBartSLTModel
from model.quantize_slt import MBartQuantizedSLTModel

import cv2

import datetime

from misc.git_utils import save_git_info
from typing import Any, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))


# NOTE: the hydra appp only inisitalize once
@hydra.main(
    # config_path="../configs", config_name="t5_text_pretrain_8a100", version_base="1.3.2"
    config_path="../configs",
    # config_name="slt_finetune_8a100",
    config_name="slt_quantized_8a100",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    train(cfg, hydra_config)


def init_output_dir(config_name: str) -> str:
    """
    Initialize the output directory for the job.
    """
    now = datetime.datetime.now()
    subfolder = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", config_name + "test", subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def init_logger(local_rank, output_dir: str):
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


def train(
    cfg: DictConfig,
    hydra_config: DictConfig,
) -> None:
    config_name = hydra_config.job.config_name

    output_dir = os.environ.get(
        config_name.upper() + "_OUTPUT_DIR",
        None,
    )
    if output_dir is None:
        print("Output directory not found in environment variables, initializing...")
        output_dir = init_output_dir(config_name)
        os.environ[config_name.upper() + "_OUTPUT_DIR"] = output_dir

    init_logger(local_rank, output_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_dir}")

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        DebugCallback(),
    ]

    cfg.data.datamodule.num_workers = 1

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=[0],
        callbacks=cbs,
        log_every_n_steps=50,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,  # NOTE: gradient clipping will be normed
        # gradient_clip_algorithm="value",
        sync_batchnorm=True,
        precision=cfg.precision,
        logger=None,
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
    # model = instantiate(cfg.model, cfg)
    # model.load_from_pretrained(
    #     "outputs/t5_text_pretrain_8a100/2025-06-18_19-56-11/epoch=79-val_generate_bleu=0.5015-blo6e98y.ckpt"
    # )
    # model = SLTModelForT5FineTune.load_from_checkpoint(cfg.ckpt, cfg=cfg)
    # model = MBartSLTModel(cfg=cfg)
    model = MBartQuantizedSLTModel(cfg=cfg)
    t.fit(model, datamodule=datamodule)


class DebugCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.current_train_batch = None
        self.logger = logging.getLogger("debug_callback")

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # for name, param in pl_module.named_parameters():
        #     self.logger.info(
        #         f"Parameter {name} - requires_grad: {param.requires_grad}, shape: {param.shape}, mean: {param.mean()}, std: {param.std()}"
        #     )
        # self.logger.info(
        #     f"Training started with model: {pl_module.__class__.__name__}, global rank: {trainer.global_rank}, local rank: {trainer.local_rank}"
        # )
        pass

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
        pass
        # # NOTE: check the loss
        # if torch.isnan(loss).any():
        #     video = self.current_train_batch["video"]
        #     ids = self.current_train_batch["ids"]
        #
        #     logger.warning(f"Loss is NaN: {loss}")
        #     logger.warning(
        #         f"Video shape: {video.shape}, mean: {video.mean()}, std: {video.std()}"
        #     )
        #     logger.warning(f"input_ids: {ids}")
        # trainer.should_stop = True

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Any,
    ) -> None:
        nan_flag = False
        if trainer.is_global_zero:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self.logger.info(
                        f"Parameter {name} - GRADE, mean: {param.grad.mean()}, std: {param.grad.std()}"
                    )

        return


if __name__ == "__main__":
    main()
