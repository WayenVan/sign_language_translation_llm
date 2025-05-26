import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from typing import List
from einops import rearrange, repeat
from torchmetrics import Accuracy, BLEUScore
from transformers import DataCollatorForWholeWordMask


class VTGHandle(BaseHandle):
    """
    Handles the model hooks for the VTM task.
    """

    def __init__(
        self,
        vocab_size,
        tokenizer,
        cfg,
    ):
        super().__init__()

        self.loss_weight = cfg.vtg_weight
        self.vocab_size = vocab_size
        self.mask_ratio = cfg.vtg_mask_ratio

        self.train_accu = Accuracy(
            task="multiclass",
            num_classes=self.vocab_size,
            ignore_index=0,  # padding index
        )

        self.bleu = BLEUScore(n_gram=1, smooth=True)
        self.collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.mask_ratio,
            mask_replace_prob=cfg.vtg_random_replace_prob,
            random_replace_prob=cfg.vtg_mask_replace_prob,
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
        module.log("train_generate_accu", train_acc, prog_bar=True, sync_dist=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self, module):
        """
        Called at the end of the validation epoch.
        """
        bleu = self.bleu.compute()
        module.log("val_generate_bleu", bleu, prog_bar=True, sync_dist=True)
        self.bleu.reset()

    def tokenize(self, text: List[str], tokenizer, device, use_mask=True):
        """
        Tokenize the text using the tokenizer.
        """
        labels_ids = []
        input_ids = []
        for setence in text:
            tokenized = tokenizer.tokenize(
                setence,
            )
            tokenized = tokenizer.convert_tokens_to_ids(tokenized)
            label = tokenized + [tokenizer.eos_token_id]
            intput = [tokenizer.bos_token_id] + tokenized

            labels_ids.append(label)
            input_ids.append(intput)

        input_outputs = tokenizer.pad(
            {"input_ids": input_ids},
            padding="longest",
            return_attention_mask=True,
        )
        if use_mask:
            input_outputs["input_ids"] = self.collator(input_ids)["input_ids"]

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
            torch.LongTensor(input_outputs["attention_mask"]).to(
                device
            ),  # (B), text length
        )

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

        # Generate causal mask
        text_casual_mask = torch.triu(
            torch.ones((L, L)),
            diagonal=1,
        ).to(device)
        text_casual_mask = torch.where(text_casual_mask == 1, 0, 1)
        casual_mask = F.pad(text_casual_mask, (0, 0, num_video_queries, 0), value=0)
        casual_mask = F.pad(casual_mask, (num_video_queries, 0, 0, 0), value=1)

        return (
            text_mask * casual_mask
        )  # (B, L + num_video_queries, L + num_video_queries)

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

        visual_features = bert_output.hidden_states[-1][:, T:, :]
        textaul_embeddings = bert_output.hidden_states[-1][:, :T, :]
        text_logits = bert_output.logits

        return visual_features, textaul_embeddings, text_logits, text_attention_mask

    def train_step(
        self, module: lightning, batch, batch_idx, visual_embeddings, v_length
    ):
        ids, _, _, text = self.dispatch_batch(batch, module.device)
        text_ids, labels, text_mask = self.tokenize(
            text, module.tokenizer, module.device, use_mask=True
        )

        out_logit = self._forward(
            module, visual_embeddings, v_length, text_ids, text_mask
        )[-2]

        # Add numerical stability
        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.train_accu.update(
            rearrange(out_loglogit, "b l c -> (b l) c"),
            rearrange(labels, "b l -> (b l)"),
        )

        target_loss = (
            F.nll_loss(
                rearrange(out_loglogit, "b l c -> (b l) c"),
                rearrange(labels, "b l -> (b l)"),
                ignore_index=0,
            )
            * self.loss_weight
        )
        module.log("train_generate_loss", target_loss, prog_bar=True)
        return target_loss

    def validation_step(
        self, module: lightning, batch, batch_idx, visual_embeddings, v_length
    ):
        ids, _, _, text = self.dispatch_batch(batch, module.device)

        B = len(ids)

        with torch.no_grad():
            # NOTE: max length in dev set is 50
            generated = module.generate(
                video_embeddings=visual_embeddings, video_length=v_length, max_length=50
            )
            generated = generated.cpu().tolist()

        for b in range(B):
            indexs = generated[b]
            try:
                eos_index = indexs.index(module.tokenizer.eos_token_id)
            except ValueError:
                eos_index = len(indexs)
            indexs = indexs[:eos_index]
            predicted = module.tokenizer.decode(indexs, skip_special_tokens=True)
            self.bleu.update(
                [predicted],
                [[text[b]]],
            )
