import sys
from tqdm import tqdm

sys.path.append(".")
from data.ph14t.ph14t_torch_dataset import Ph14KeywordDataset
from data.ph14t.ph14t_lightning_datamodule import Ph14TDataModule
from hydra import compose, initialize
from hydra.utils import instantiate


def test_dataset():
    data_root = "/root/projects/slt_set_llms/dataset/PHOENIX-2014-T-release-v3"
    keyword_dir = "/root/projects/slt_set_llms/outputs/keywords"
    mode = "train"

    dataset = Ph14KeywordDataset(data_root, keyword_dir, mode)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        print(data["keywords"])


def test_datamodule():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("test_train")
    with open(cfg.data.vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    datamodule = Ph14TDataModule(cfg)
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    for batch in tqdm(train_dataloader):
        print(batch["video"].shape)
        # print(batch["translation"])
        pass


if __name__ == "__main__":
    # test_dataset()
    test_datamodule()
