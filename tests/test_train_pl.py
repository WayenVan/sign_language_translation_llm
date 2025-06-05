import sys
import os
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # NOTE: this is the initial cwd when runing the sciprt

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer

from lightning import Trainer
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch

from model.slt_pl import SLTModelForLLMFineTune
import cv2

from misc.git_utils import save_git_info
from typing import Any, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)  # NOTE: hydra already setupo the logger for us
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error


# NOTE: the hydra appp only inisitalize once
@hydra.main(
    config_path="../configs", config_name="prompt_learning_l40s", version_base="1.3.2"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


def train(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_config.runtime.output_dir
    config_name = hydra_config.job.config_name

    logger.info(f"Output directory: {output_dir}")

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        DebugCallback(),
    ]

    cfg.data.datamodule.num_workers = 1
    cfg.data.batch_size = 1

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
        precision="bf16-mixed",
        logger=None,
        # WARN: will slow down the training process, just for debug now
        # detect_anomaly=True,
        num_sanity_val_steps=2,  # NOTE: disable sanity check
    )

    logger.info(f"Process in local rank {t.local_rank}, global rank {t.global_rank}")

    datamodule = instantiate(cfg.data.datamodule, cfg)

    model = SLTModelForLLMFineTune(cfg=cfg)
    state_dict = torch.load(
        cfg.pretrained_checkpoint, map_location=f"cuda:{t.local_rank}"
    )["state_dict"]
    model.load_from_bootstrap(state_dict)

    # start train
    t.fit(model, datamodule=datamodule)

    t.fit(model, datamodule=datamodule)


class DebugCallback(callbacks.Callback):
    @staticmethod
    def check_nan_hook(module, input, output):
        if isinstance(output, tuple):  # 有的模块输出是 tuple
            output = output[0]
        if not torch.isnan(output).any():
            return
        logging.warning(
            f"NaN detected in module: {module.__class__.__name__} ({module})"
        )

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pass
        # hooks = []
        # for name, module in pl_module.named_modules():
        #     hooks.append(module.register_forward_hook(self.check_nan_hook))
        #
        # return super().on_train_start(trainer, pl_module)

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

            logging.info(
                f"Param: Param {name} has  mean: {param.mean()}, std: {param.std()}"
            )

            if param.grad is not None:
                logging.info(
                    f"Grade: Param {name} has grad mean: {param.grad.mean()}, std: {param.grad.std()}"
                )
            else:
                logging.info(f"Param {name} has no grad")
        if trainer.global_step > 100:
            trainer.should_stop = True

        return


if __name__ == "__main__":
    main()
