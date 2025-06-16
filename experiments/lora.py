import torch
from torch import nn
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

from peft import get_peft_model, LoraConfig, TaskType

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

# image = Image.open("outputs/visualization_val/0.jpg")
# image = image.resize((192, 256))
# image = image.convert("RGB")
# image = np.array(image, dtype=np.float32) / 255.0
# image = F.to_tensor(image)
# inputs = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
# inputs = inputs.unsqueeze(0)  # Add batch dimension


image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-plus-base", device_map=device
)

target_modules = []
for name, module in model.named_modules():
    target_modules.append(name)
    print(name, module.__class__.__name__)


peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=target_modules,
)

model = get_peft_model(model, peft_config)
trainable, all = model.get_nb_trainable_parameters()

# for name, param in model.named_parameters():
#     print(name)

print(
    f"Trainable parameters: {trainable}, All parameters: {all}, Ratio: {trainable / all:.2%}"
)


# with torch.no_grad():
#     outputs = model(
#         inputs,
#         dataset_index=torch.LongTensor([0]),
#         output_hidden_states=True,  # 0 is the best
#     )  # coco whole body
#
# import matplotlib.pyplot as plt
#
# for i in range(17):
#     heatmap = outputs.heatmaps[0, i].cpu().numpy()
#
#     plt.imshow(heatmap)
#     plt.savefig("outputs/viz_pose/{}.jpg".format(i))
