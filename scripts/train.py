import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), "../"))
)  # NOTE: this is the initial cwd when runing the sciprt, the hydra will change the cwd to the output dir

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.cuda.amp.grad_scaler import GradScaler

from lightning import Trainer
from lightning.pytorch import plugins
from lightning.pytorch import callbacks

from model.slt import SLTModel


# NOTE: the hydra appp only inisitalize once
@hydra.main(config_path="../configs", config_name="test_train", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    train(cfg)


def train(cfg: DictConfig) -> None:
    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory: {working_dir}")

    # NOTE: load vocab
    with open(cfg.data.vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        callbacks.LearningRateMonitor("step", log_momentum=True),
        callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="epoch={epoch:02d}-wer={vall_accu:.2f}",
            monitor="val_wer",
            mode="max",
            save_last=True,
        ),
    ]

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=getattr(cfg, "devices", "auto"),
        callbacks=cbs,
        log_every_n_steps=50,
        max_epochs=50,
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        plugins=[
            plugins.MixedPrecision(
                precision="16-mixed",
                device="cuda",
                scaler=GradScaler(
                    growth_interval=100,
                ),
            ),
        ],
    )

    datamodule = instantiate(cfg.data.datamodule, cfg)
    model = SLTModel(cfg, vocab)
    t.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
