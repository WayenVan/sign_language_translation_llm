import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from torchmetrics import Accuracy
from einops import rearrange


class MLMHandle(BaseHandle):
    """
    Handles the model hooks for the MLM task.
    """

    def __init__(self, vocab_size, loss_weight, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

        self.train_accu = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.val_accu = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.loss_weight = loss_weight

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

    @staticmethod
    def mask_tokens(
        input_ids, tokenizer, mlm_prob=0.15, mask_prob=0.8, random_prob=0.1
    ):
        labels = input_ids.clone()
        special = torch.tensor(
            [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],
            device=input_ids.device,
        )
        maskable = ~((input_ids.unsqueeze(-1) == special).any(-1))
        probs = (
            torch.full(input_ids.shape, mlm_prob, device=input_ids.device)
            * maskable.float()
        )
        masked = torch.bernoulli(probs).bool()
        labels[~masked] = -100

        # 80%替换成 [MASK]
        indices_replaced = (
            torch.bernoulli(
                torch.full(input_ids.shape, mask_prob, device=input_ids.device)
            ).bool()
            & masked
        )
        input_ids[indices_replaced] = tokenizer.mask_token_id

        # 10%替换成随机token
        indices_random = (
            torch.bernoulli(
                torch.full(
                    input_ids.shape,
                    random_prob / (1 - mask_prob),
                    device=input_ids.device,
                )
            ).bool()
            & masked
            & ~indices_replaced
        )
        input_ids[indices_random] = torch.randint(
            tokenizer.vocab_size, input_ids.shape, device=input_ids.device
        )[indices_random]

        # 剩下的保持原token
        return input_ids, labels

    def _forward(self, module, video, video_length, text):
        tokenizer_outputs = module.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            # add_special_tokens=False,  # NOTE: <CLs> and <SEP> will be added by the model
        )
        text_ids = tokenizer_outputs["input_ids"].to(module.device)  # (B, L)
        text_length = tokenizer_outputs["attention_mask"].sum(1).to(module.device)

        with torch.no_grad():
            masked_text_ids, mask_text_labels = self.mask_tokens(
                text_ids,
                module.tokenizer,
                mlm_prob=self.mask_ratio,
            )

        with torch.no_grad():
            visual_encoder_outputs = module.visual_encoder(video, video_length)
        hidden_state = visual_encoder_outputs.hidden_state
        v_length = visual_encoder_outputs.video_length
        visual_embeddings = module.visual_adapter(hidden_state)  # b t c

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

        out_features = module.shared_encoder(
            inputs_embeds=features,
            attention_mask=padding_attention_mask,
        )
        out_features = out_features.last_hidden_state[:, T:, :]
        out_logit = module.shared_encoder_header(out_features)  # b l vocab_size
        return out_logit, mask_text_labels

    def train_step(self, module, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        out_logit, mask_text_labels = self._forward(module, video, video_length, text)

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.train_accu.update(
            rearrange(out_loglogit, "b l c -> (b l) c"),
            rearrange(mask_text_labels, "b l -> (b l)"),
        )

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

        out_logit, mask_text_labels = self._forward(module, video, video_length, text)

        out_loglogit = F.log_softmax(out_logit, dim=-1)
        self.val_accu.update(
            rearrange(out_loglogit, "b l c -> (b l) c"),
            rearrange(mask_text_labels, "b l -> (b l)"),
        )


if __name__ == "__main__":
    mask = MLMHandle.generate_attention_mask(torch.tensor([5, 3]), torch.tensor([4, 2]))
    print(mask)
