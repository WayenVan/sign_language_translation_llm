import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from torchmetrics import Accuracy
from einops import rearrange, repeat
from transformers import DataCollatorForWholeWordMask


class VTMHandle(BaseHandle):
    """
    Handles the model hooks for the VTM task.
    """

    def __init__(self, tokenizer, vocab_size, cfg):
        super().__init__()
        self.mask_ratio = cfg.vtm_mask_ratio

        self.train_accu = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.val_accu = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.loss_weight = cfg.vtm_weight
        self.collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.mask_ratio,
            random_replace_prob=cfg.vtm_random_replace_prob,
            mask_replace_prob=cfg.vtm_mask_replace_prob,
        )

    def dispatch_batch(self, batch, device):
        ids = batch["id"]
        video = batch["video"].to(device)
        video_length = batch["video_length"].to(device)
        text = batch["text"]

        return ids, video, video_length, text

    def on_train_epoch_end(self, module):
        """
        Called at the end of the training epoch.
        """
        train_acc = self.train_accu.compute()
        module.log("train_masked_accu", train_acc, prog_bar=True, sync_dist=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self, module):
        """
        Called at the end of the validation epoch.
        """
        val_acc = self.val_accu.compute()
        module.log("val_masked_accu", val_acc, prog_bar=True, sync_dist=True)
        self.val_accu.reset()

    @staticmethod
    def generate_padding_attention_mask(
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

        text_mask = repeat(text_attention_mask, "b l -> b ll l", b=B, ll=L)
        text_mask = F.pad(text_mask, (0, 0, num_video_queries, 0), value=0)
        text_mask = F.pad(text_mask, (num_video_queries, 0), value=1)
        # (B, L+num_video_queries, L + num_video_queries)

        return text_mask  # (B, L + num_video_queries, L + num_video_queries)

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

    def tokenize(self, module, text, device):
        tokenizer_outputs = module.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # NOTE: <CLs> and <SEP> will be added by the model
        )
        text_ids = tokenizer_outputs["input_ids"].to(module.device)  # (B, L)
        text_attention_mask = tokenizer_outputs["attention_mask"].to(
            module.device
        )  # (B, L)

        # NOTE: masking the text token by data collator
        masked_output = self.collator(text_ids.cpu().tolist())
        masked_text_ids, mask_text_labels = (
            masked_output["input_ids"].to(module.device),
            masked_output["labels"].to(module.device),
        )
        return masked_text_ids, mask_text_labels, text_attention_mask

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
        padding_attention_casual = self.generate_padding_attention_mask(
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
        textaul_embeddings = bert_output.hidden_states[-1][:, -L:, :]
        text_logits = bert_output.logits

        return visual_features, textaul_embeddings, text_logits, text_attention_mask

    def train_step(
        self, module: lightning, batch, batch_idx, visual_embeddings, v_length
    ):
        ids, _, _, text = self.dispatch_batch(batch, module.device)

        masked_text_ids, mask_text_labels, text_attention_mask = self.tokenize(
            module, text, module.device
        )
        _, _, out_logit, _ = self._forward(
            module, visual_embeddings, v_length, masked_text_ids, text_attention_mask
        )

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.train_accu.update(
            rearrange(out_loglogit, "b l c -> (b l) c"),
            rearrange(mask_text_labels, "b l -> (b l)"),
        )

        # WARN: in case all labels are -100, the loss will be 0, impossible situation
        # because the mask_token will reproduce if no token is masked
        if (mask_text_labels == -100).all():
            loss = torch.tensor(0.0, device=out_loglogit.device)
        else:
            loss = (
                F.nll_loss(
                    rearrange(out_loglogit, "b l c -> (b l) c"),
                    rearrange(mask_text_labels, "b l -> (b l)"),
                    ignore_index=-100,
                )
                * self.loss_weight
            )

        module.log("train_masked_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, module, batch, batch_idx, visual_embeddings, v_length):
        ids, _, _, text = self.dispatch_batch(batch, module.device)

        masked_text_ids, mask_text_labels, text_attention_mask = self.tokenize(
            module, text, module.device
        )
        _, _, out_logit, _ = self._forward(
            module, visual_embeddings, v_length, masked_text_ids, text_attention_mask
        )

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.val_accu.update(
            rearrange(out_logit, "b l c -> (b l) c"),
            rearrange(mask_text_labels, "b l -> (b l)"),
        )


if __name__ == "__main__":
    mask = VTMHandle.generate_attention_mask(torch.tensor([5, 3]), torch.tensor([4, 2]))
    print(mask)
