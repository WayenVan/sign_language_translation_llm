from numpy import ma
import torch
from torch import nn
from lightning import LightningModule
from transformers import AutoTokenizer
from omegaconf import DictConfig
from transformers.models.mistral3 import Mistral3ForConditionalGeneration
from hydra.utils import instantiate
from torchmetrics import Accuracy
from torch.optim import Optimizer
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us


class SLTModel(LightningModule):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        # write only for this model
        self.llm_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        self.vocab = vocab
        self.reverse_vocab = {word: i for i, word in enumerate(vocab)}
        self._create_tokenizer()
        self._create_llm_embedding_layer()
        self._create_vocab_convert_mapping()
        self.padding_idx = self.reverse_vocab[self.llm_tokenizer.pad_token]
        self.bos_idx = self.reverse_vocab[self.llm_tokenizer.bos_token]
        self.eos_idx = self.reverse_vocab[self.llm_tokenizer.eos_token]

        self.visual_backbone = instantiate(cfg.visual_backbone)
        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(
            cfg.decoder,
            padding_idx=self.padding_idx,
            bos_token_id=self.bos_idx,
            eos_token_id=self.eos_idx,
            vocab_size=len(vocab),
        )
        self.loss = instantiate(cfg.loss)

        self.train_accu = Accuracy(
            task="multiclass", num_classes=len(vocab), ignore_index=self.padding_idx
        )
        self.val_accu = Accuracy(
            task="multiclass", num_classes=len(vocab), ignore_index=self.padding_idx
        )

    def _create_tokenizer(self):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
        self.llm_tokenizer.padding_side = "right"

    def _create_llm_embedding_layer(self):
        model = Mistral3ForConditionalGeneration.from_pretrained(
            self.llm_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        ).eval()

        self.llm_embedding_layer = model.get_input_embeddings()
        for param in self.llm_embedding_layer.parameters():
            param.requires_grad = False
        self.llm_hidden_size = self.llm_embedding_layer.embedding_dim
        del model

    def _pad_tokens_for_decoder(self, tokens_ids):
        """
        @param tokens_ids: list of int tensor, token ids
        """
        device = tokens_ids[0].device

        # Pad the tokens to the maximum length
        nt = torch.nested.nested_tensor(tokens_ids)
        padded = torch.nested.to_padded_tensor(nt, padding=self.padding_idx)

        attnion_mask = padded.ne(self.padding_idx).to(torch.int64)
        lenghts = attnion_mask.sum(dim=-1)
        return padded, attnion_mask, lenghts

    def _create_vocab_convert_mapping(self):
        mapping = []  # index: llm_id -> my_vocab_id
        for id_local in range(len(self.vocab)):
            token = self.vocab[id_local]
            id_llm = self.llm_tokenizer.convert_tokens_to_ids(token)
            mapping.append(id_llm)

        self.register_buffer(
            "vocab_mapping_local_to_llm",
            torch.tensor(mapping, dtype=torch.int64),
            persistent=False,
        )

    def sentence_to_ids(self, sentence, add_bos=False, add_eos=False):
        """
        Convert a sentence to a list of token IDs.
        """
        tokens = self.llm_tokenizer.tokenize(sentence)
        if add_bos:
            tokens = [self.llm_tokenizer.bos_token] + tokens

        if add_eos:
            tokens = tokens + [self.llm_tokenizer.eos_token]

        ids = [self.reverse_vocab[token] for token in tokens]

        return ids

    def forward(
        self,
        video,
        video_length=None,
        max_length=None,
    ):
        """
        Forward pass through the model.
        """

        v_feats, v_length = self.visual_backbone(video, video_length)
        v_feats, v_length, video_padding_mask = self.encoder(v_feats, v_length)
        decoder_outputs = self.decoder(
            visual_hidden_states=v_feats,
            video_padding_mask=video_padding_mask,
            max_length=max_length,
        )
        return decoder_outputs

    def preprocess_train_keywords(self, keywords):
        keywords_ids_in = [
            torch.tensor(
                self.sentence_to_ids(k, add_bos=True, add_eos=False),
                dtype=torch.int64,
                device=self.device,
            )
            for k in keywords
        ]

        keywords_ids_out = [
            torch.tensor(
                self.sentence_to_ids(k, add_bos=False, add_eos=True),
                dtype=torch.int64,
                device=self.device,
            )
            for k in keywords
        ]

        keywords_ids_in, mask, keywords_lengths = self._pad_tokens_for_decoder(
            keywords_ids_in
        )
        keywords_ids_out, _mask, _ = self._pad_tokens_for_decoder(keywords_ids_out)

        keywords_llm_in = self.llm_tokenizer(
            keywords, return_tensors="pt", padding=True
        )

        # assert the valid range of tokens between llms and my decoder should be the same
        for i in range(len(keywords_ids_in)):
            assert mask[i].cpu().tolist() == keywords_llm_in[i].attention_mask
            assert mask[i].cpu().tolist() == _mask[i].cpu().tolist()

        return (
            keywords_ids_in,
            keywords_llm_in,
            keywords_ids_out,
            mask,
            keywords_lengths,
        )

    def training_step(self, batch, batch_idx):
        keywords = batch["keywords"]
        video = batch["video"].to(self.device)
        video_length = batch["video_length"].to(self.device)

        (
            keywords_ids_in,
            keywords_llm_in,
            keywords_ids_out,
            mask,
            keywords_lengths,
        ) = self.preprocess_train_keywords(keywords)

        v_feats, v_length = self.visual_backbone(video, video_length)
        v_feats, v_length, video_padding_mask = self.encoder(v_feats, v_length)
        decoder_outputs = self.decoder(
            input_ids=keywords_ids_in,
            visual_hidden_states=v_feats,
            attention_mask=mask,
            video_padding_mask=video_padding_mask,
        )

        target_token_llm_features = self.llm_embedding_layer(
            keywords_llm_in.input_ids.to(self.device)
        )
        loss_outputs = self.loss(
            decoder_outputs,
            target_token_llm_features,
            keywords_ids_out,
            mask,
            self.padding_idx,
        )

        # Calculate the logits and target IDs for token-level accuracy
        self.train_accu.update(
            decoder_outputs.logits.flatten(0, 1), keywords_ids_out.flatten()
        )

        # Log the loss
        for loss_name in loss_outputs._asdict():
            self.log(loss_name, getattr(loss_outputs, loss_name), prog_bar=True)

        if self.debug:
            logger.info(f"keywords_ids_in: {keywords_ids_in.cpu().tolist()}")
            logger.info(f"keywords_ids_out: {keywords_ids_out.cpu().tolist()}")

        return loss_outputs.loss

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        # Calculate and log the accuracy
        train_acc = self.train_accu.compute()
        self.log("train_token_level_accu", train_acc, prog_bar=True)
        self.train_accu.reset()

    def validation_step(self, batch, batch_idx):
        # teaching forceing when evalute the metrics
        ids = batch["ids"]
        keywords = batch["keywords"]
        video = batch["video"].to(self.device)
        video_length = batch["video_length"].to(self.device)

        (
            keywords_ids_in,
            keywords_llm_in,
            keywords_ids_out,
            mask,
            keywords_lengths,
        ) = self.preprocess_train_keywords(keywords)

        v_feats, v_length = self.visual_backbone(video, video_length)
        v_feats, v_length, video_padding_mask = self.encoder(v_feats, v_length)
        decoder_outputs = self.decoder(
            input_ids=keywords_ids_in,
            visual_hidden_states=v_feats,
            attention_mask=mask,
            video_padding_mask=video_padding_mask,
        )

        target_token_llm_features = self.llm_embedding_layer(
            keywords_llm_in.input_ids.to(self.device)
        )
        # NOTE: we don't need to calculate the loss in validation step, but maybe we need?
        # loss_outputs = self.loss(
        #     decoder_outputs,
        #     target_token_llm_features,
        #     keywords_ids_out,
        #     mask,
        #     self.padding_idx,
        # )

        # Calculate the logits and target IDs for token-level accuracy
        self.val_accu.update(
            decoder_outputs.logits.flatten(0, 1), keywords_ids_out.flatten()
        )

    def on_validation_epoch_end(self):
        val_acc = self.val_accu.compute()
        self.log("val_token_level_accu", val_acc, prog_bar=True)
        self.val_accu.reset()

    def train(self, is_train):
        """
        Override the train method to set the model to training mode.
        """
        super().train(is_train)
        self.llm_embedding_layer.eval()  # always eval for llm embedding layer

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.engine.optimizer,
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.visual_backbone.parameters()
                    )
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.encoder.parameters()
                    )
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.decoder.parameters()
                    )
                },
            ],
        )
        scheduler = instantiate(self.cfg.engine.lr_scheduler, opt)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, state_dict: Dict[str, Any]) -> None:
        for key in state_dict:
            if key.startswith("llm_embedding_layer"):
                del state_dict[key]


if __name__ == "__main__":
    import polars as pl

    with open("outputs/keywords_vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    cfg = {
        "debug": True,
    }
    cfg = DictConfig(cfg)
    model = SLTModel(cfg, vocab).cuda()
    # print(model.llm_embedding_layer)
    # print(model.llm_hidden_size)
    # print(model.padding_idx)

    df = pl.read_csv("outputs/keywords/train-extracted-keywords.csv", separator="|")

    idx = []
    for keywords in df["keywords"]:
        idx.append(keywords)
        if len(idx) > 10:
            break
    (
        keyword_ids_in,
        keyword_llm_in,
        keyword_ids_out,
        attention_mask,
        keywords_lengths,
    ) = model.preprocess_train_keywords(idx)

    converted = []
    for id in keyword_ids_in[0].cpu().tolist():
        converted.append(model.vocab_mapping_local_to_llm[id].item())

    print(keyword_llm_in[0].ids)
