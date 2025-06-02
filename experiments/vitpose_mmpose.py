import torch
import requests
import numpy as np

from PIL import Image
from torchvision.transforms import functional as F
from transformers.models.vitpose_backbone.modeling_vitpose_backbone import (
    VitPoseBackbone,
)

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
from mmpose.apis import init_model
import sys

sys.path.append(".")
from modules.vitpose_mmpose_encoder.vitpose_mmpose_encoder import MMVitPoseEncoder

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

image = Image.open("outputs/visualization_val/0.jpg")
image = image.resize((192, 256))
image = image.convert("RGB")
image = np.array(image, dtype=np.float32) / 255.0
image = F.to_tensor(image)
inputs = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
inputs = inputs.unsqueeze(0)  # Add batch dimension
inputs = inputs.to(device)

model = init_model(
    "vitpose_configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_base_wholebody_256x192.py",
    checkpoint="outputs/vitpose_base.pth",
    device=device,
)

with torch.no_grad():
    outputs = model(
        inputs,
        # return_loss=False,
    )  # coco whole body

import matplotlib.pyplot as plt


for i in range(17):
    heatmap = outputs[0, i].cpu().numpy()

    plt.imshow(heatmap)
    plt.savefig("outputs/viz_pose/{}.jpg".format(i))
