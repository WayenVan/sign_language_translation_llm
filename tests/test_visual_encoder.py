import torch
from einops import repeat
import sys

sys.path.append(".")

from modules.timm_visual_encoder.timm_visual_encoder import TimmVisualEncoder

model = TimmVisualEncoder("vit_base_patch16_224", 224, 512).cuda()

visual = torch.randn(2, 30, 3, 224, 224).cuda()

output = model(visual)
print(output[0].shape)
