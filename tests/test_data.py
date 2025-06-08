import sys
from tqdm import tqdm

sys.path.append(".")
from data.ph14t.ph14t_torch_dataset import Ph14TDataset
from data.ph14t.ph14t_lightning_datamodule import Ph14TDataModule
from data.ph14t.ph14t_index import Ph14TIndex
from hydra import compose, initialize
from hydra.utils import instantiate
from transformers import AutoTokenizer


def test_max_data_length():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-europeana-cased")
    data_root = "/root/shared-data/sign_language_translation_llm/dataset/PHOENIX-2014-T-release-v3/"
    ph14t_index = Ph14TIndex(data_root, mode="dev")
    max_length = 0
    for id in ph14t_index.ids:
        data_info = ph14t_index.get_data_info_by_id(id)
        text = data_info["translation"]
        lentgh = len(tokenizer.tokenize(text))
        if lentgh > max_length:
            max_length = lentgh
    print(max_length)


def test_dataset():
    data_root = "dataset/PHOENIX-2014-T-release-v3"
    mode = "train"

    dataset = Ph14TDataset(data_root, mode)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        print(data["video"].shape)


def test_datamodule():
    initialize(config_path="../configs")
    cfg = compose("initial_train_l40s")

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
    cfg = compose("initial_train_home")

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
            f"outputs/visualization_val/{i}.jpg",
            f.astype("uint8"),
        )


if __name__ == "__main__":
    # test_dataset()
    test_datamodule()
    # test_data_validation()
    # test_max_data_length()
