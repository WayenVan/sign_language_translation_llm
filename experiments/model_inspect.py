import os
import sys

sys.path.append(".")


import torch


ckpt = torch.load("epoch=epoch=05-wer=val_token_level_accu=0.49.ckpt")


for k, v in ckpt["state_dict"].items():
    print(k)
