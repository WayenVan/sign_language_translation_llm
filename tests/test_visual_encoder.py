import torch
import numpy as np
from einops import repeat, rearrange
import sys
from PIL import Image
from torchvision.transforms import functional as F

sys.path.append(".")

from modules.timm_visual_encoder.timm_visual_encoder import TimmVisualEncoder
from modules.vitpose_visual_encoder.vitpose_visual_encoder import VitPoseVisualEncoder


def test_timm_encoder():
    model = TimmVisualEncoder("vit_base_patch16_224", 224, 512).cuda()

    visual = torch.randn(2, 30, 3, 224, 224).cuda()

    output = model(visual)
    print(output[0].shape)


def test_vitpose_encoder():
    image = Image.open("outputs/visualization/0.jpg")
    image = image.resize((192, 256))
    image = image.convert("RGB")
    image = np.array(image, dtype=np.float32) / 255.0
    image = F.to_tensor(image)
    inputs = F.normalize(
        image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True
    )
    inputs = rearrange(inputs, "c h w -> 1 1 c h w")

    model = VitPoseVisualEncoder("usyd-community/vitpose-plus-small")
    with torch.no_grad():
        hidden_states, video_length, _ = model(inputs, [1])
    print(hidden_states.shape)
    print(video_length)
    import matplotlib.pyplot as plt

    for i in range(17):
        heatmap = _[0, 0, i].cpu().numpy()

        plt.imshow(heatmap)
        plt.savefig("outputs/heatmap_{}.jpg".format(i))


if __name__ == "__main__":
    test_vitpose_encoder()
