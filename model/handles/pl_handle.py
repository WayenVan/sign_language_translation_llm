from .base_handle import BaseHandle
import torch
from torchmetrics import Accuracy
from torchmetrics.text import BLEUScore
import torch.nn.functional as F
from typing import List
from einops import rearrange
import logging

logger = logging.getLogger(__name__)  # NOTE: lightning already setup the logger for us


class PLHandle(BaseHandle):
    """
    Handles the model hooks for the SLT task.
    Actually is very similar to the itg handle
    """

    def __init__(self, module, vocab_size, loss_weight, llm_padding_idx):
        super().__init__()
        self.loss_weight = loss_weight
        self.vocab_size = vocab_size

        self.train_accu = Accuracy(
            task="multiclass",
            num_classes=self.vocab_size,
            ignore_index=llm_padding_idx,  # padding index
        )
        self.bleu = BLEUScore(n_gram=1, smooth=True)
        self.llm_padding_idx = llm_padding_idx

        # NOTE: freeze adapter, and shared encoder
        # for param in module.visual_adapter.parameters():
        #     param.requires_grad = False
        # for param in module.shared_encoder.parameters():
        #     param.requires_grad = False
        # module.visual_adapter.eval()
        # module.shared_encoder.eval()

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
        module.log("train_llm_generate_accu", train_acc, prog_bar=True, sync_dist=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self, module):
        """
        Called at the end of the validation epoch.
        """
        bleu = self.bleu.compute()
        module.log("val_llm_generate_bleu", bleu, prog_bar=True, sync_dist=True)
        self.bleu.reset()

    @staticmethod
    def tokenize(text: List[str], tokenizer, device):
        """
        Tokenize the text using the tokenizer.
        """
        eos_token_id = tokenizer.encoder["</s>"]
        bos_token_id = tokenizer.encoder["</s>"]

        text_ids = [tokenizer.encode(t, add_special_tokens=False) for t in text]
        input_ids = [[bos_token_id] + text_id for text_id in text_ids]
        label_ids = [text_id + [eos_token_id] for text_id in text_ids]

        input_outputs = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        label_outputs = tokenizer.pad(
            {"input_ids": label_ids},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return (
            torch.LongTensor(input_outputs["input_ids"]).to(device),
            torch.LongTensor(label_outputs["input_ids"]).to(device),
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

    def _forward_q_former(
        self,
        module,
        visual_features,  # [b t c]
        v_length,  # [b]
        output_attentions=False,
    ):
        B, T, C = visual_features.shape
        NUM_QUERIES = module.num_query_token

        # video query
        video_query_tokens = module.video_query_tokens.expand(B, -1, -1)

        # create padding mask for the cross atttention with video
        cross_attention_mask = self.length_to_mask(v_length, max_length=T)

        # create dummy text ids
        dummy_text_ids = (
            torch.LongTensor([module.tokenizer.pad_token_id])
            .to(module.device)
            .expand(B, 1)
        )
        dummy_attention_mask = torch.ones(B, NUM_QUERIES + 1).to(module.device).long()
        dummy_attention_mask[:, -1] = 0  # first token is padding

        # q-former forward
        bert_output = module.shared_encoder(
            input_ids=dummy_text_ids,  # dummy input to avoid using text_ids
            attention_mask=dummy_attention_mask,
            query_embeds=video_query_tokens,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=cross_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        visual_features = bert_output.hidden_states[-1][:, :-1]
        visual_features = module.connector(visual_features)

        assert visual_features.shape[1] == NUM_QUERIES
        return visual_features

    def _forward_llm(self, module, visual_features, text_ids, text_attention_mask):
        B, NUM_QUERIES, C = visual_features.shape

        eos_token_id = module.llm_tokenizer.encoder["</s>"]
        eos_token_embedding = module.llm.get_input_embeddings()(
            torch.LongTensor([eos_token_id]).to(module.device)
        )

        llm_encode_features = torch.cat(
            [
                visual_features,
                eos_token_embedding.expand(B, 1, C),
            ],
            dim=1,  # [b, num_queries+1, C]
        )

        llm_outputs = module.llm(
            inputs_embeds=llm_encode_features,
            decoder_input_ids=text_ids,
            decoder_attention_mask=text_attention_mask,
            use_cache=False,
        )
        return llm_outputs.logits

    def train_step(self, module, batch, batch_idx, visual_embeddings, v_length):
        ids, _, _, text = self.dispatch_batch(batch, module.device)
        text_ids, labels, text_mask = self.tokenize(
            text, module.llm_tokenizer, module.device
        )
        visual_features = self._forward_q_former(module, visual_embeddings, v_length)
        out_logit = self._forward_llm(module, visual_features, text_ids, text_mask)

        if out_logit.isnan().any():
            logger.warning(
                f"NaN detected, ids: {ids}, text: {text}, visual_features:{visual_features.mean()}, out_logit:{out_logit.mean()}"
            )

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
        module.log("train_llm_generate_loss", target_loss, prog_bar=True)
        return target_loss

    def validation_step(self, module, batch, batch_idx, visual_embeddings, v_length):
        ids, _, _, text = self.dispatch_batch(batch, module.device)

        text_ids, labels, text_mask = self.tokenize(
            text,
            module.llm_tokenizer,
            module.device,
        )

        visual_features = self._forward_q_former(module, visual_embeddings, v_length)

        B, NUM_QUERIES, C = visual_features.shape

        eos_token_id = module.llm_tokenizer.encoder["</s>"]
        eos_token_embedding = module.llm.get_input_embeddings()(
            torch.LongTensor([eos_token_id]).to(module.device)
        )

        llm_encode_features = torch.cat(
            [
                visual_features,
                eos_token_embedding.expand(B, 1, C),
            ],
            dim=1,  # [b, num_queries+1, C]
        )
        with torch.no_grad():
            # NOTE: max length in dev set is 50
            outputs = module.llm.generate(
                inputs_embeds=llm_encode_features,
                max_length=50,
            )

        for b in range(B):
            predicted = module.llm_tokenizer.decode(
                outputs[b], skip_special_tokens=True
            )
            self.bleu.update(
                [predicted],
                [[text[b]]],
            )
