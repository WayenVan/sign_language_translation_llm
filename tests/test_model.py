from pydoc import visiblename
import sys

sys.path.append(".")
from model.slt import SLTModel
from data.ph14t import Ph14TDataModule
from hydra import compose, initialize
import torch


def test_slt_model_inspect():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("initial_train")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    model = SLTModel(
        cfg=cfg,
    )
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)


def test_slt_model():
    import polars as pl

    initialize(config_path="../configs")
    cfg = compose("initial_train")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    model = SLTModel(
        cfg=cfg,
    ).cuda()
    loader = data_module.train_dataloader()
    for i, batch in enumerate(loader):
        with torch.autocast("cuda"):
            model.training_step(batch, 0)
            # model.validation_step(batch, 0)
            print("ok")


def test_slt_model_generation():
    initialize(config_path="../configs")
    cfg = compose("initial_train")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    # model = SLTModel.load_from_checkpoint(
    #     "outputs/train/2025-06-08_15-07-26/05hhj2d9-epoch=61-val_generate_bleu=0.2160.ckpt",
    #     strict=False,
    #     cfg=cfg,
    # )
    model = SLTModel(
        cfg=cfg,
    ).cuda()
    loader = data_module.train_dataloader()
    for i, batch in enumerate(loader):
        if i < 8:
            continue
        ids = batch["id"]
        video = batch["video"].to(model.device)
        video_length = batch["video_length"].to(model.device)
        text = batch["text"]
        text_ids = model.tokenizer(text)
        # print(text_ids)
        print(text)

        # Generate
        generated_ids = model.generate(video, video_length, max_length=20)

        for item in generated_ids.cpu().tolist():
            print(model.tokenizer.decode(item, skip_special_tokens=True))

        break


if __name__ == "__main__":
    # test_slt_model()
    # test_slt_model_inspect()
    test_slt_model_generation()
