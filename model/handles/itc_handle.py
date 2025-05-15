import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from typing import List
from einops import rearrange
from torchmetrics import Accuracy


class ITCHandle(BaseHandle):
    """
    Handles the model hooks for the MLM task.
    """

    def __init__(
        self,
        vocab_size,
        loss_weight,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.vocab_size = vocab_size

        self.train_accu = Accuracy(
            task="multiclass", num_classes=self.vocab_size, ignore_index=0
        )

        self.val_accu = Accuracy(
            task="multiclass", num_classes=self.vocab_size, ignore_index=0
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
        self.log("train_generate_accu", train_acc, prog_bar=True, sync_dist=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self, module):
        """
        Called at the end of the validation epoch.
        """
        val_acc = self.val_accu.compute()
        module.log("val_generate_accu", val_acc, prog_bar=True, sync_dist=True)
        self.val_accu.reset()

    @staticmethod
    def tokenize(text: List[str], tokenizer, device):
        """
        Tokenize the text using the tokenizer.
        """
        labels_ids = []
        intput_ids = []
        for setence in text:
            tokenized = tokenizer.tokenize(
                setence,
            )
            tokenized = tokenizer.convert_tokens_to_ids(tokenized)
            label = tokenized + [tokenizer.sep_token_id]
            intput = [tokenizer.cls_token_id] + tokenized
            labels_ids.append(label)
            intput_ids.append(intput)

        input_outputs = tokenizer.pad(
            {"input_ids": intput_ids},
            padding="longest",
            return_attention_mask=True,
        )
        label_outputs = tokenizer.pad(
            {"input_ids": labels_ids},
            padding="longest",
            return_attention_mask=True,
        )

        assert input_outputs["attention_mask"] == label_outputs["attention_mask"], (
            "Attention mask should be the same for input and label"
        )

        return (
            torch.LongTensor(input_outputs["input_ids"]).to(device),
            torch.LongTensor(label_outputs["input_ids"]).to(device),
            torch.LongTensor(input_outputs["attention_mask"])
            .sum(dim=1)
            .to(device),  # (B), text length
        )

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

        video_mask = video_mask.long()
        text_mask = text_mask.long()

        padding_mask = torch.cat((video_mask, text_mask), dim=1).unsqueeze(
            1
        )  # for heads dimension and query dimension # (B,  1, L)

        # Generate causal mask
        causal_mask = torch.triu(
            torch.ones((max_text_length, max_text_length)),
            diagonal=1,
        ).to(video_mask.device)
        causal_mask = torch.where(causal_mask == 1, 0, 1)
        casual_mask = F.pad(
            causal_mask, (max_video_length, 0, max_video_length, 0), value=1
        )
        return padding_mask * casual_mask

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

        # 5. 生成注意力掩码
        padding_attention_casual = self.generate_padding_casual_attention_mask(
            v_length, text_length
        )
        features = torch.cat((visual_embeddings, textaul_embeddings), dim=1)  # b t+l c

        out_features = module.shared_encoder(
            inputs_embeds=features,
            attention_mask=padding_attention_casual,
        )
        out_features = out_features.last_hidden_state[:, T:, :]
        out_logit = module.shared_encoder_header(out_features)  # b l vocab_size
        return out_logit

    def train_step(self, module: lightning, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        text_ids, labels, text_length = self.tokenize(
            text, module.tokenizer, module.device
        )

        out_logit = self._forward(module, video, video_length, text_ids, text_length)

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.train_accu.update(
            rearrange(out_loglogit, "b l c -> (b l) c"),
            rearrange(labels, "b l -> (b l)"),
        )

        loss = (
            F.nll_loss(
                rearrange(out_loglogit, "b l c -> (b l) c"),
                rearrange(labels, "b l -> (b l)"),
            )
            * self.loss_weight
        )
        module.log(
            "train_generate_loss",
            loss,
            prog_bar=True,
        )
        return loss

    def validation_step(self, module: lightning, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, module.device)

        text_ids, labels, text_length = self.tokenize(
            text, module.tokenizer, module.device
        )

        out_logit = self._forward(module, video, video_length, text_ids, text_length)

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.val_accu.update(
            rearrange(out_loglogit, "b l c -> (b l) c"),
            rearrange(labels, "b l -> (b l)"),
        )
