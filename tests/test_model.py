import sys

sys.path.append(".")
from model.slt import SLTModel
from data.ph14t import Ph14TDataModule
from hydra import compose, initialize
import torch


def test_slt_model():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("prompt_learning")
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


def test_slt_model_generation():
    initialize(config_path="../configs")
    cfg = compose("initial_train")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    model = SLTModel.load_from_checkpoint(
        "/root/projects/slt_set_llms/outputs/train/2025-05-18_19-58-41/epoch=epoch=55-wer=val_generate_accu=0.76.ckpt",
        strict=False,
        cfg=cfg,
    )
    loader = data_module.train_dataloader()
    for i, batch in enumerate(loader):
        if i < 8:
            continue
        ids = batch["ids"]
        video = batch["video"].to(model.device)
        video_length = batch["video_length"].to(model.device)
        text = batch["text"]
        print(text)

        # Generate
        generated_ids = model.generate(video, video_length, 20)
        print(generated_ids)

        break


if __name__ == "__main__":
    # test_slt_model()
    test_slt_model_generation()
