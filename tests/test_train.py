import os
import sys
import numpy as np

sys.path.append(".")


def test_train():
    import polars as pl
    from model.slt import SLTModel
    from hydra import compose, initialize
    import torch
    from lightning import Trainer
    from lightning.pytorch import plugins
    from torch.cuda.amp.grad_scaler import GradScaler
    from hydra.utils import instantiate
    import cv2

    cv2.setNumThreads(0)

    initialize(config_path="../configs")
    cfg = compose("test_train")

    with open("outputs/keywords_vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    t = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0, 1],
        # devices=[0],
        callbacks=[],
        log_every_n_steps=50,
        max_epochs=50,
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
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
    test_train()
