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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

image = Image.open("outputs/visualization/0.jpg")
image = image.resize((192, 256))
image = image.convert("RGB")
image = np.array(image, dtype=np.float32) / 255.0
image = F.to_tensor(image)
inputs = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
inputs = inputs.unsqueeze(0)  # Add batch dimension


image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-small")
model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-plus-small", device_map=device
)


with torch.no_grad():
    outputs = model(
        inputs,
        dataset_index=torch.LongTensor([0]),
        output_hidden_states=True,  # 0 is the best
    )  # coco whole body

import matplotlib.pyplot as plt

for i in range(17):
    heatmap = outputs.heatmaps[0, i].cpu().numpy()

    plt.imshow(heatmap)
    plt.savefig("outputs/visualization/heatmap_{}.jpg".format(i))
