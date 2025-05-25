import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from torchmetrics import Accuracy
from einops import rearrange
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
        ids = batch["ids"]
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

        video_mask = video_mask.long()
        text_mask = text_mask.long()
        return torch.cat((video_mask, text_mask), dim=1).unsqueeze(
            1
        )  # for heads dimension and query dimension # (B,  1, L)

    def _forward(self, module, video, video_length, text, output_attentions=False):
        tokenizer_outputs = module.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # NOTE: <CLs> and <SEP> will be added by the model
        )
        text_ids = tokenizer_outputs["input_ids"].to(module.device)  # (B, L)
        text_length = tokenizer_outputs["attention_mask"].sum(1).to(module.device)

        # NOTE: masking the text token by data collator
        masked_output = self.collator(text_ids.cpu().tolist())
        masked_text_ids, mask_text_labels = (
            masked_output["input_ids"].to(module.device),
            masked_output["labels"].to(module.device),
        )

        with torch.no_grad():
            visual_encoder_outputs = module.visual_encoder(video, video_length)
        hidden_state = visual_encoder_outputs.hidden_state
        v_length = visual_encoder_outputs.video_length
        visual_embeddings, v_length = module.visual_adapter(
            hidden_state, v_length
        )  # b t c

        with torch.no_grad():
            textaul_embeddings = module.shared_encoder.get_input_embeddings()(
                masked_text_ids
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

        bert_output = module.shared_encoder(
            inputs_embeds=features,
            attention_mask=padding_attention_mask,
            output_attentions=output_attentions,
        )
        out_features = bert_output.last_hidden_state[:, T:, :]
        out_logit = module.shared_encoder_header(out_features)  # b l vocab_size
        return out_logit, mask_text_labels, bert_output.attentions

    def train_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        out_logit, mask_text_labels, _ = self._forward(
            module, video, video_length, text
        )

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        # Add NaN check
        if torch.isnan(out_loglogit).any():
            raise ValueError("NaN detected in log probabilities")

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

    def validation_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        out_logit, mask_text_labels, _ = self._forward(
            module, video, video_length, text
        )

        self.val_accu.update(
            rearrange(out_logit, "b l c -> (b l) c"),
            rearrange(mask_text_labels, "b l -> (b l)"),
        )


if __name__ == "__main__":
    mask = VTMHandle.generate_attention_mask(torch.tensor([5, 3]), torch.tensor([4, 2]))
    print(mask)
