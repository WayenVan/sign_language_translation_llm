from mmpose.apis import init_model

from PIL import Image
from torchvision.transforms import functional as F
import torch

import matplotlib.pyplot as plt


model = init_model(
    "sapiens_pose_configs/sapiens_pose/coco_wholebody/sapiens_0.6b-210e_coco_wholebody-1024x768.py",
    checkpoint="outputs/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178.pth?download=true",
    device="cpu",
)

model.cuda()
model.eval()

for name, param in model.named_parameters():
    print(name, param.size())

# input_image = Image.open("/root/shared-data/Radar_Yao/outputs/video/video_20.png")
input_image = Image.open("outputs/visualization_val/110.jpg")
input_image = input_image.convert("RGB")
input_image = F.resize(input_image, (1024, 768), antialias=True)

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
    output = torch.sigmoid(output)


for i in range(28):
    plt.imshow(output[0][i].cpu().numpy(), vmin=0.0, vmax=1.0)
    plt.savefig(f"outputs/sapiens_pose/{i}.jpg")
