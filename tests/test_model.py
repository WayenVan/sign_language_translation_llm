from pydoc import visiblename
import sys

sys.path.append(".")
from model.slt import SLTModel
from model.slt_vision_pretrain import SignBackboneForVPretraining
from model.t5_text_pretrain import ModelForT5TextPretrain
from model.mbart_slt import MBartSLTModel
from model.quantize_slt import MBartQuantizedSLTModel
from data.ph14t import Ph14TDataModule
from hydra import compose, initialize
import torch
from hydra.utils import instantiate


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
    cfg = compose("slt_quantized_8a100")
    cfg.data.batch_size = 2
    data_module = Ph14TDataModule(cfg)
    data_module.setup()
    # model = instantiate(cfg.model, cfg).to("cuda:0")
    # model = ModelForT5TextPretrain(
    #     cfg=cfg,
    # ).to("cuda:0")
    # model.load_from_pretrained(
    #     "outputs/t5_text_pretrain_8a100/2025-06-18_19-56-11/epoch=79-val_generate_bleu=0.5015-blo6e98y.ckpt"
    # )
    model = MBartQuantizedSLTModel(cfg).to("cuda:0")
    loader = data_module.train_dataloader()
    for i, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
    ).to("cuda:4")
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
    test_slt_model()
    # test_slt_model_inspect()
    # test_slt_model_generation()
