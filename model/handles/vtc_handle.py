import torch
from .base_handle import BaseHandle
from misc.clip_loss import clip_loss
from typing import List
from misc.circular_queue import CircularQueue
from torch.nn import functional as F
from lightning.pytorch import LightningModule

from einops import rearrange, einsum


class VTCHandle(BaseHandle):
    """
    Handles the model hooks for the VTM task.
    """

    def __init__(self, hiddent_size, cfg):
        super().__init__()
        self.loss_weight = cfg.vtc_weight
        self.fine_grain_loss_weight = cfg.vtc_fine_grain_weight
        self.hiddent_size = hiddent_size
        self.temperature_fine_grain = cfg.vtc_temperature_fine_grain
        self.top_k = cfg.vtc_top_k
        # self.queue_max_size = cfg.vtc_queue_max_size
        # self.queue_initialized = False

    # def _initiate_queue(self, device, dtype):
    #     self.visual_queue = CircularQueue(
    #         self.queue_max_size, self.hiddent_size, device, dtype
    #     )
    #     self.text_queue = CircularQueue(
    #         self.queue_max_size, self.hiddent_size, device, dtype
    #     )
    #     self.queue_initialized = True

    def dispatch_batch(self, batch, device):
        ids = batch["id"]
        video = batch["video"].to(device)
        video_length = batch["video_length"].to(device)
        text = batch["text"]

        return ids, video, video_length, text

    def on_train_epoch_end(self, module):
        """
        do not clean the queue
        """
        # self.visual_queue.reset()
        # self.text_queue.reset()
        pass

    @staticmethod
    def generate_padding_casual_attention_mask(
        num_video_queries,
        text_attention_mask,  # [B, L]
    ):
        """
        Generate addtivie attention mask for the video and text sequences.
        video_length: [B]
        text_length: [B]
        """
        B, L = text_attention_mask.shape
        device = text_attention_mask.device

        text_mask = F.pad(text_attention_mask, (num_video_queries, 0), value=1)
        text_mask = text_mask.unsqueeze(1)  # (B, 1, L + num_video_queries)

        mask = torch.ones(
            (B, L + num_video_queries, L + num_video_queries), device=device
        )

        mask[:, num_video_queries:, :num_video_queries] = 0
        mask[:, :num_video_queries, num_video_queries:] = 0

        return text_mask * mask  # (B, L + num_video_queries, L + num_video_queries)

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
            torch.LongTensor(input_outputs["attention_mask"]).to(
                device
            ),  # (B), text length
        )

    @staticmethod
    def length_to_mask(lengths, max_length=None):
        """
        Convert lengths to a boolean mask.
        lengths: [B]
        max_length: int, optional
        """
        if max_length is None:
            max_length = lengths.max().item()
        B = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(
            B, max_length
        ) < lengths.unsqueeze(1)
        return mask.long()  # (B, max_length)

    def _forward(
        self,
        module,
        visual_features,  # [b t c]
        v_length,  # [b]
        text_ids,  # [B, L]
        text_attention_mask,  # [b l]
        output_attentions=False,
    ):
        B, T, C = visual_features.shape
        B, L = text_ids.shape

        # create mask for shared encoder
        padding_attention_casual = self.generate_padding_casual_attention_mask(
            module.num_query_token, text_attention_mask
        )

        # create padding mask for the cross atttention with video
        cross_attention_mask = self.length_to_mask(v_length, max_length=T)

        # video query
        video_query_tokens = module.video_query_tokens.expand(B, -1, -1)

        # q-former forward
        bert_output = module.shared_encoder(
            input_ids=text_ids,
            query_embeds=video_query_tokens,
            attention_mask=padding_attention_casual,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=cross_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        visual_features = bert_output.hidden_states[-1][:, :-L, :]
        textaul_embeddings = bert_output.hidden_states[-1][:, -L, :]
        textauL_embedding_fine_grain = bert_output.hidden_states[-1][:, -L + 1 :, :]
        text_logits = bert_output.logits

        return (
            visual_features,
            textaul_embeddings,
            text_logits,
            text_attention_mask,
            textauL_embedding_fine_grain,
        )

    def contrastive_loss(
        self, module: LightningModule, visual_features, text_features, label_smooth=0.1
    ):
        """
        visual_features: [B, T, C]
        text_features: [B,  C]
        """
        current_rank = module.global_rank
        B, T, C = visual_features.shape

        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        global_text_features = module.all_gather(text_features)
        global_text_features = rearrange(global_text_features, "w b c -> (w b) c")

        v2t = einsum(
            visual_features,
            global_text_features,
            "b t c, wb c -> b t wb ",
        )
        v2t = v2t.max(dim=-2).values

        # ------------------ text to visual features ------------------

        global_visual_features = module.all_gather(visual_features)  # [w B T C]
        t2v = einsum(
            text_features,
            global_visual_features,
            "b c, w bv t c -> w b bv t",
        )
        t2v = t2v.max(dim=-1).values
        t2v = rearrange(t2v, "w b bv -> b (w bv)")

        # ------------------ gather the target ------------------

        target = torch.arange(
            current_rank * B, current_rank * B + B, device=visual_features.device
        )

        if t2v.shape[-1] == 1:
            assert v2t.shape[-1] == 1, "Both should be 1 if one is 1"
            # WARN: if there is only one video, we cannot use cross entropy loss
            # use bce loss instead
            loss = (
                F.binary_cross_entropy_with_logits(v2t, torch.ones_like(v2t).float())
                + F.binary_cross_entropy_with_logits(t2v, torch.ones_like(t2v).float())
            ) / 2.0
        else:
            loss = (
                F.cross_entropy(v2t, target, label_smoothing=label_smooth)
                + F.cross_entropy(t2v, target, label_smoothing=label_smooth)
            ) / 2.0
        return loss

    def fine_grain_contrastive(self, video, video_length, text, text_mask):
        """
        Compute contrastive loss between from video to text, the text is detached
        Args:
            video: Video embeddings of shape (B, T, D)
            video_length: Lengths of video sequences of shape (B,)
            text: Text embeddings of shape (B, L, D)
            text_mask: Attention mask for text of shape (B, L)
        """

        video_feats = torch.nn.functional.normalize(video, dim=-1, p=2)
        text_feats = torch.nn.functional.normalize(text, dim=-1, p=2).detach()
        video_mask = self.length_to_mask(
            video_length, max_length=video_feats.size(1)
        )  # (B, T)
        padding_mask = video_mask.unsqueeze(2)  # (B, T, L)
        padding_mask = padding_mask.float()  # Convert to float for masking
        padding_mask = padding_mask.masked_fill(
            padding_mask == 0, float("-inf")
        )  # Set padding to -inf
        padding_mask = padding_mask.masked_fill(
            padding_mask == 1, 0.0
        )  # Set non-padding to 0.0

        similarity = (
            torch.bmm(video_feats, text_feats.transpose(1, 2))
            / self.temperature_fine_grain
        )  # (B, T, L)
        similarity = similarity + padding_mask  # Apply padding mask
        similarity = torch.log_softmax(similarity, dim=-2)  # Log softmax

        value, indeces = torch.topk(
            similarity, k=self.top_k, dim=-2
        )  # Get top-k indices, (B, 5, L)
        mask = torch.zeros_like(similarity, dtype=torch.float)  # (B, T, L)
        mask.scatter_(-2, indeces, 1.0)  # Set top-k indices to 1
        mask = mask * text_mask.unsqueeze(1).float()  # Apply text mask

        loss = -(similarity * mask).sum(dim=-2).mean()  # Compute loss
        return loss

    def train_step(self, module, batch, batch_idx, visual_embeddings, v_length):
        ids, _, _, text = self.dispatch_batch(batch, module.device)
        text_ids, text_mask = self.tokenize(text, module.tokenizer, module.device)

        visual_features, text_features, _, _, text_fine_grain_features = self._forward(
            module, visual_embeddings, v_length, text_ids, text_mask
        )

        target_loss = (
            self.contrastive_loss(module, visual_features, text_features, 0.1)
            * self.loss_weight
        )
        module.log("train_contrastive_loss", target_loss, prog_bar=True)

        fine_grain_loss = (
            self.fine_grain_contrastive(
                visual_embeddings,
                v_length,
                text_fine_grain_features,
                text_mask[:, 1:],  # remove cls token
            )
            * self.fine_grain_loss_weight
        )
        module.log("train_fine_grain_loss", fine_grain_loss, prog_bar=True)

        target_loss += fine_grain_loss
        return target_loss

    def validation_step(self, module, batch, batch_idx, visual_embeddings, v_length):
        pass
