import sys

sys.path.append(".")
from model.slt import SLTModel
from hydra import compose, initialize
import torch


def test_slt_model():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("test_train")
    model = SLTModel(cfg).cuda()

    df = pl.read_csv("outputs/keywords/train-extracted-keywords.csv", separator="|")

    kws = []
    for keywords in df["keywords"]:
        kws.append(keywords)
        if len(kws) >= 2:
            break

    batch = {
        "ids": torch.tensor([0, 1]).cuda(),
        "names": kws,
        "keywords": kws,
        "video": torch.randn(2, 20, 3, 224, 224).cuda(),
        "video_length": torch.tensor([20, 16]).cuda(),
    }
    engines = model.configure_optimizers()

    model.training_step(batch, 0)
    model.validation_step(batch, 0)


if __name__ == "__main__":
    test_slt_model()
