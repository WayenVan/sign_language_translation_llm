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

from model.slt import SLTModel
import cv2

from misc.git_utils import save_git_info

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error


# NOTE: the hydra appp only inisitalize once
@hydra.main(config_path="../configs", config_name="test_train", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    train(cfg)


def train(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    working_dir = hydra_config.runtime.output_dir
    config_name = hydra_config.job.config_name

    logger.info(f"Output directory: {working_dir}")

    # NOTE: load vocab
    with open(cfg.data.vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        callbacks.LearningRateMonitor("step", log_momentum=True),
        callbacks.ModelCheckpoint(
            dirpath=working_dir,
            filename="epoch={epoch:02d}-wer={val_token_level_accu:.2f}",
            monitor="val_token_level_accu",
            mode="max",
            save_last=True,
        ),
    ]

    # NOTE: set the logger
    lt_logger = WandbLogger(
        name=config_name,
        project="sign-langauge-translation-llm",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=getattr(cfg, "devices", "auto"),
        callbacks=cbs,
        log_every_n_steps=50,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=0.5,  # NOTE: gradient clipping will be normed
        sync_batchnorm=True,
        precision="16-mixed",
        logger=lt_logger,
    )

    if t.is_global_zero:
        # NOTE: save git info
        save_git_info(
            repo_path=project_root,
            info_path=os.path.join(working_dir, "git_info"),
        )

    logger.info(f"Process in local rank {t.local_rank}, global rank {t.global_rank}")

    datamodule = instantiate(cfg.data.datamodule, cfg)
    model = SLTModel(cfg, vocab)
    t.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
