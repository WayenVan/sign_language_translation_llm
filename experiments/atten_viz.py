import sys

from torch.nn import attention

sys.path.append(".")
from model.slt import SLTModel
from data.ph14t import Ph14TDataModule
from hydra import compose, initialize
import torch

import seaborn as sns
import matplotlib.pyplot as plt

initialize(config_path="../configs")
cfg = compose("initial_train")
data_module = Ph14TDataModule(cfg)
data_module.setup()
model = SLTModel.load_from_checkpoint(
    "/root/projects/slt_set_llms/outputs/train/2025-05-20_23-18-04/last.ckpt",
    cfg=cfg,
).cuda()
loader = data_module.train_dataloader()

# handle = model.handles["vtg"]
# for batch in loader:
#     with torch.no_grad():
#         ids, video, video_length, text = handle.dispatch_batch(batch, model.device)
#         text_ids, labels, text_length = handle.tokenize(
#             text, model.tokenizer, model.device
#         )
#         _, atten = handle._forward(
#             model, video, video_length, text_ids, text_length, output_attentions=True
#         )
#     break
#
# # atten = list([batch, heads, from sequence, to sequence])
# atten = atten[-1][0].mean(dim=0)
# atten = atten[128:, :]
# atten = atten.cpu().numpy()
#
# sns.heatmap(atten, cmap="viridis")
# plt.tight_layout()
# plt.savefig("outputs/atten.png")

# --------------------------------------------------------
#
handle = model.handles["vtg"]
for batch in loader:
    with torch.no_grad():
        ids, video, video_length, text = handle.dispatch_batch(batch, model.device)
        text_ids, labels, text_length = handle.tokenize(
            text, model.tokenizer, model.device, use_mask=True
        )
        out_logit, atten = handle._forward(
            model,
            video,
            video_length,
            text_ids,
            text_length,
            output_attentions=True,
        )
    break

# atten = list([batch, heads, from sequence, to sequence])
atten = atten[-1][0].mean(dim=0)
# atten = atten[128:, :]
atten = atten.cpu().numpy()

sns.heatmap(atten, cmap="viridis")
plt.tight_layout()
plt.savefig("outputs/atten_mask.png")
