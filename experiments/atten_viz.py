import sys

from torch.nn import attention

sys.path.append(".")
from model.slt import SLTModel
from data.ph14t import Ph14TDataModule
from hydra import compose, initialize
import torch
from omegaconf import OmegaConf

import seaborn as sns
import matplotlib.pyplot as plt

initialize(config_path="../configs")

cfg = compose("initial_train")
data_module = Ph14TDataModule(cfg)
data_module.setup()

model_cfg = OmegaConf.load("outputs/train/2025-06-09_16-42-39/.hydra/config.yaml")
model = SLTModel.load_from_checkpoint(
    "outputs/train/2025-06-09_16-42-39/iftai1w1-epoch=72-val_generate_bleu=0.2172.ckpt",
    cfg=model_cfg,
).cuda()
loader = data_module.val_dataloader()

for i, batch in enumerate(loader):
    generated = model.generate(
        batch["video"].cuda(),
        batch["video_length"].cuda(),
        max_length=50,
        num_beams=5,
    )
    decoded = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
    # decoded = [d[10:] for d in decoded]  # Skip the <s> token
    print("Original:", batch["text"])
    print("Generated:", decoded)
