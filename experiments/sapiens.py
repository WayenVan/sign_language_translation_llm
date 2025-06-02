import sys

sys.path.append(".")
from misc import hack_registry
from mmpose.apis import init_model

from PIL import Image
import torchvision.transforms.functional as F
import torch
from transformers.generation.utils import GenerationMixin
from mmpretrain.models.backbones.vision_transformer import VisionTransformer

from mmengine.config import Config
import matplotlib.pyplot as plt

from einops import rearrange
import numpy as np


model = init_model(
    "sapeins_configs/sapiens_pose/coco_wholebody/sapiens_0.3b-210e_coco_wholebody-1024x768.py",
    checkpoint="outputs/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth",
    device="cpu",
)

model.cuda()
model.eval()

for name, param in model.named_parameters():
    print(name, param.size())

# input_image = Image.open("/root/shared-data/Radar_Yao/outputs/video/video_20.png")
input_image = Image.open("outputs/visualization_val/110.jpg")
input_image = input_image.convert("RGB")
input_image = F.resize(input_image, (256, 192), antialias=True)

input_tensor = F.to_tensor(input_image) * 255.0
input_tensor = F.normalize(
    input_tensor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)
input_batch = input_tensor.unsqueeze(
    0
).cuda()  # create a mini-batch as expected by the model


with torch.no_grad():
    output = model(input_batch)


for i in range(133):
    plt.imshow(output[0][i].cpu().numpy(), vmin=0.0, vmax=1.0)
    plt.savefig(f"outputs/sapiens_pose/{i}.jpg")
