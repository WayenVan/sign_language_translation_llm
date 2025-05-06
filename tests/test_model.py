import sys

sys.path.append(".")
from model.slt import SLTModel
from omegaconf import DictConfig
from hydra import compose, initialize


def test_slt_model():
    import polars as pl

    with open("outputs/keywords_vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    initialize(config_path="../configs")
    cfg = compose("test_train")
    model = SLTModel(cfg, vocab).cuda()

    df = pl.read_csv("outputs/keywords/train-extracted-keywords.csv", separator="|")

    idx = []
    for keywords in df["keywords"]:
        idx.append(keywords)
        if len(idx) > 10:
            break


if __name__ == "__main__":
    test_slt_model()
