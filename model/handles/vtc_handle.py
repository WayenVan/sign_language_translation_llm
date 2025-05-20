import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from torchmetrics import Accuracy
from einops import rearrange, repeat
from misc.contrastive_loss import masked_bi_directional_contrastive_loss


class VTCHandle(BaseHandle):
    """
    Handles the model hooks for the VTM task.
    """

    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def dispatch_batch(self, batch, device):
        ids = batch["ids"]
        video = batch["video"].to(device)
        video_length = batch["video_length"].to(device)
        text = batch["text"]

        return ids, video, video_length, text

    @staticmethod
    def generate_padding_attention_mask(
        video_length, text_length, max_video_length=None, max_text_length=None
    ):
        """
        Generate addtivie attention mask for the video and text sequences.
        video_length: [B]
        text_length: [B]
        """
        B = video_length.size(0)
        assert video_length.size(0) == text_length.size(0), (
            f"video length {video_length.size(0)} does not match text length {text_length.size(0)}"
        )

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

        video_mask = video_mask.long()
        text_mask = text_mask.long()
        padding_mask = torch.cat((video_mask, text_mask), dim=1).unsqueeze(
            1
        )  # for heads dimension and query dimension # (B,  1, L)   kkkk

        forbidden_mask = torch.ones(
            B, max_video_length + max_text_length, max_video_length + max_text_length
        ).to(video_length.device)

        forbidden_mask[:, max_video_length:, :max_video_length] = 0
        forbidden_mask[:, :max_video_length, max_video_length:] = 0

        return padding_mask * forbidden_mask

    def _forward(self, module, video, video_length, text):
        tokenizer_outputs = module.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # NOTE: <CLs> and <SEP> is deleted
        )
        text_ids = tokenizer_outputs["input_ids"].to(module.device)  # (B, L)
        text_length = tokenizer_outputs["attention_mask"].sum(1).to(module.device)

        with torch.no_grad():
            visual_encoder_outputs = module.visual_encoder(video, video_length)
        hidden_state = visual_encoder_outputs.hidden_state
        v_length = visual_encoder_outputs.video_length
        visual_embeddings = module.visual_adapter(hidden_state)  # b t c

        with torch.no_grad():
            textaul_embeddings = module.shared_encoder.get_input_embeddings()(
                text_ids
            )  # b l c

        B, T, C = visual_embeddings.shape
        _, L, _ = textaul_embeddings.shape

        assert T == v_length.max().item(), (
            f"Visual length {T} does not match max video length {v_length.max().item()}"
        )
        assert L == text_length.max().item(), (
            f"Text length {L} does not match max text length {text_length.max().item()}"
        )

        # 5. 生成注意力掩码
        padding_attention_mask = self.generate_padding_attention_mask(
            v_length, text_length
        )
        features = torch.cat((visual_embeddings, textaul_embeddings), dim=1)  # b t+l c

        out_features = module.shared_encoder(
            inputs_embeds=features,
            attention_mask=padding_attention_mask,
        )
        text_out_features = out_features.last_hidden_state[:, T:, :]
        visual_out_features = out_features.last_hidden_state[:, :T, :]

        return text_out_features, visual_out_features, text_length

    def train_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        text_out_features, visual_out_features, text_length = self._forward(
            module, video, video_length, text
        )

        # WARN: in case all labels are -100, the loss will be 0, impossible situation
        # because the mask_token will reproduce if no token is masked
        loss = (
            masked_bi_directional_contrastive_loss(
                visual_out_features, text_out_features, video_length, text_length
            )
            * self.loss_weight
        )

        module.log("train_contrastive_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        text_out_features, visual_out_features, text_length = self._forward(
            module, video, video_length, text
        )

        with torch.no_grad():
            # WARN: in case all labels are -100, the loss will be 0, impossible situation
            # because the mask_token will reproduce if no token is masked
            loss = (
                masked_bi_directional_contrastive_loss(
                    visual_out_features, text_out_features, video_length, text_length
                )
                * self.loss_weight
            )

        module.log("val_contrastive_loss", loss, prog_bar=True)
