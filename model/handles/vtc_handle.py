import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from torchmetrics import Accuracy
from einops import rearrange, repeat
from misc.clip_loss import clip_loss
from typing import List, override
from misc.circular_queue import CircularQueue


class VTCHandle(BaseHandle):
    """
    Handles the model hooks for the VTM task.
    """

    def __init__(self, hiddent_size, cfg):
        super().__init__()
        self.loss_weight = cfg.vtc_weight
        self.queue_max_size = cfg.vtc_queue_max_size
        self.hiddent_size = hiddent_size
        self.queue_initialized = False

    def _initiate_queue(self, device, dtype):
        self.visual_queue = CircularQueue(
            self.queue_max_size, self.hiddent_size, device, dtype
        )
        self.text_queue = CircularQueue(
            self.queue_max_size, self.hiddent_size, device, dtype
        )
        self.queue_initialized = True

    def dispatch_batch(self, batch, device):
        ids = batch["ids"]
        video = batch["video"].to(device)
        video_length = batch["video_length"].to(device)
        text = batch["text"]

        return ids, video, video_length, text

    def on_train_epoch_end(self, module):
        """
        clean the queue after each epoch
        """
        self.visual_queue.reset()
        self.text_queue.reset()

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

    def tokenize(self, text: List[str], tokenizer, device):
        """
        Tokenize the text using the tokenizer.
        """
        input_ids = []
        for setence in text:
            tokenized = tokenizer.tokenize(
                setence,
            )
            tokenized = tokenizer.convert_tokens_to_ids(tokenized)
            intput = [tokenizer.cls_token_id] + tokenized

            input_ids.append(intput)

        input_outputs = tokenizer.pad(
            {"input_ids": input_ids},
            padding="longest",
            return_attention_mask=True,
        )

        return (
            torch.LongTensor(input_outputs["input_ids"]).to(device),
            torch.LongTensor(input_outputs["attention_mask"])
            .sum(dim=1)
            .to(device),  # (B), text length
        )

    def _forward(self, module, video, video_length, text_ids, text_length):
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

        # NOTE: add visual cls token to the model
        visual_embeddings = torch.cat(
            (module.video_cls_token.expand(B, -1, -1), visual_embeddings), dim=1
        )
        v_length = v_length + 1

        # 5. 生成注意力掩码
        padding_attention_mask = self.generate_padding_attention_mask(
            v_length, text_length
        )
        features = torch.cat((visual_embeddings, textaul_embeddings), dim=1)  # b t+l c

        out_features = module.shared_encoder(
            inputs_embeds=features,
            attention_mask=padding_attention_mask,
        )
        text_cls_feature = out_features.last_hidden_state[:, T, :]
        visual_cls_feature = out_features.last_hidden_state[:, 0, :]

        return visual_cls_feature, text_cls_feature, text_length

    def train_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)
        text_ids, text_length = self.tokenize(text, module.tokenizer, module.device)
        text_out_features, visual_out_features, text_length = self._forward(
            module, video, video_length, text_ids, text_length
        )

        # initialize the queue if not already done
        if not self.queue_initialized:
            self._initiate_queue(module.device, visual_out_features.dtype)

        # NOTE: get cached viual text pair from the queue
        total_visual_logits = torch.cat(
            [visual_out_features, self.visual_queue.get_queue()], dim=0
        )
        total_text_logits = torch.cat(
            [text_out_features, self.text_queue.get_queue()], dim=0
        )

        # WARN: in case all labels are -100, the loss will be 0, impossible situation
        # because the mask_token will reproduce if no token is masked
        loss = (
            clip_loss(
                total_visual_logits,
                total_text_logits,
                module.contrastive_logit_scale,
            )
            * self.loss_weight
        )

        module.log("train_contrastive_loss", loss, prog_bar=True)

        with torch.no_grad():
            # NOTE: all gather all other features
            gather_visual_feature = module.all_gather(visual_out_features)
            gather_text_feature = module.all_gather(text_out_features)

            # NOTE: put the pairs into the queue
            self.visual_queue.enqueue(
                torch.cat(gather_visual_feature, dim=0).detach().contiguous()
            )
            self.text_queue.enqueue(
                torch.cat(gather_text_feature, dim=0).detach().contiguous()
            )

        return loss

    def validation_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)
        text_ids, text_length = self.tokenize(text, module.tokenizer, module.device)
        text_out_features, visual_out_features, text_length = self._forward(
            module, video, video_length, text_ids, text_length
        )

        with torch.no_grad():
            # WARN: in case all labels are -100, the loss will be 0, impossible situation
            # because the mask_token will reproduce if no token is masked
            loss = (
                clip_loss(
                    visual_out_features,
                    text_out_features,
                    module.contrastive_logit_scale,
                )
                * self.loss_weight
            )

        module.log("val_contrastive_loss", loss, prog_bar=True)
