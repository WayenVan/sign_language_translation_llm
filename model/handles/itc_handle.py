import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F


class ITCHandle(BaseHandle):
    """
    Handles the model hooks for the MLM task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dispatch_batch(self, batch):
        ids = batch["ids"]
        video = batch["video"]
        video_length = batch["video_length"]
        text = batch["text"]
        text_length = batch["text_length"]

        return ids, video, video_length, text, text_length

    @staticmethod
    def generate_padding_casual_attention_mask(
        video_length, text_length, max_video_length=None, max_text_length=None
    ):
        """
        Generate addtivie attention mask for the video and text sequences.
        video_length: [B]
        text_length: [B]
        """
        if max_video_length is None:
            max_video_length = video_length.max().item()
        if max_text_length is None:
            max_text_length = text_length.max().item()

        video_mask = torch.arange(max_video_length, device=video_length.device).expand(
            video_length.size(0), max_video_length
        ) < video_length.unsqueeze(1)
        text_mask = torch.arange(max_text_length, device=text_length.device).expand(
            text_length.size(0), max_text_length
        ) < text_length.unsqueeze(1)

        video_mask = torch.where(video_mask, False, float("-inf"))
        text_mask = torch.where(text_mask, False, float("-inf"))
        padding_mask = (
            torch.cat((video_mask, text_mask), dim=1).unsqueeze(1).unsqueeze(2)
        )  # for heads dimension and query dimension # (B, 1, 1, L)

        # Generate causal mask
        causal_mask = torch.triu(
            torch.ones((max_text_length, max_text_length)),
            diagonal=1,
        ).to(video_mask.device)
        causal_mask = torch.where(causal_mask == 1, float("-inf"), 0.0)
        casual_mask = F.pad(
            causal_mask, (max_video_length, 0, max_video_length, 0), value=0
        )
        return padding_mask + casual_mask

    def train_step(self, module: lightning, batch, batch_idx):
        ids, video, video_length, text, text_length = self.dispatch_batch(batch)
        v, v_length = module.encoder()
