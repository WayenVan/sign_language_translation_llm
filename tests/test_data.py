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
    initialize(config_path="../configs")
    cfg = compose("test_train")

    datamodule = Ph14TDataModule(cfg)
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    for batch in tqdm(train_dataloader):
        print(batch["video"].shape)
        # print(batch["translation"])
        pass


def test_data_validation():
    import cv2

    initialize(config_path="../configs")
    cfg = compose("test_train")

    del cfg.data.transforms.train.transforms[-2]
    del cfg.data.transforms.val.transforms[-2]
    cfg.data.batch_size = 2

    datamodule = Ph14TDataModule(cfg)
    datamodule.setup("fit")
    train_dataloader = datamodule.val_dataloader()
    for batch in tqdm(train_dataloader):
        video = batch["video"][0]
        break

    video = video.cpu().numpy().transpose(0, 2, 3, 1) * 255
    for i in range(video.shape[0]):
        f = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"/root/projects/slt_set_llms/outputs/visualization_val/{i}.jpg",
            f.astype("uint8"),
        )


if __name__ == "__main__":
    # test_dataset()
    # test_datamodule()
    test_data_validation()
