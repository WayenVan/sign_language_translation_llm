import torch
from lightning import LightningModule
from transformers import AutoTokenizer, AutoConfig
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Dict, Any
import logging
from typing import Optional, List
from einops import rearrange
from collections import OrderedDict

from transformers.models.bert.modeling_bert import BertModel, BertConfig
from .handles.itc_handle import ITCHandle
from .handles.mlm_handle import MLMHandle

from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us


class SLTModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug

        # write only for this model

        self._create_layers()
        self._create_bert_shared_encoder()
        self._create_handles()

    def _create_handles(self):
        self.itc_flag = getattr(self.cfg, "itc_flag", False)
        self.mlm_flag = getattr(self.cfg, "mlm_flag", False)
        self.prompt_learning = getattr(self.cfg, "prompt_learning", False)

        self.handles = nn.ModuleDict()

        if self.itc_flag:
            self.handles["itc"] = ITCHandle(self.vocab_size, self.cfg.itc_weight)
            self.itc_weight = self.cfg.itc_weight

        if self.mlm_flag:
            self.handles["mlm"] = MLMHandle(
                self.vocab_size,
                self.cfg.mlm_mask_ratio,
            )
            self.mlm_weight = self.cfg.mlm_weight

        if self.prompt_learning and (self.mlm_flag or self.itc_flag):
            raise ValueError(
                "Prompt learning is not supported with MLM or ITC. Please set prompt_learning to False."
            )

        if self.prompt_learning:
            self.handles["prompt_learning"]  # TODO: implement prompt learning

    def _create_layers(self):
        self.visual_encoder = instantiate(self.cfg.modules.visual_encoder)
        self.visual_adapter = instantiate(self.cfg.modules.visual_adapter)

        # NOTE: visual bacbone is frozen
        for paras in self.visual_encoder.parameters():
            paras.requires_grad = False
        self.visual_encoder.eval()

    def _create_bert_shared_encoder(self):
        self.shared_encoder = BertModel.from_pretrained(
            self.cfg.modules.bert_shared_encoder_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.modules.bert_shared_encoder_id,
            use_fast=True,
        )
        self.tokenizer.padding_side = "right"

        self.bert_config = AutoConfig.from_pretrained(
            self.cfg.modules.bert_shared_encoder_id,
        )
        self.vocab_size = self.bert_config.vocab_size
        self.shared_encoder_header = nn.Linear(
            self.bert_config.hidden_size,
            self.vocab_size,
        )

        # NOTE: freeze all the embedding model in the layer
        for paras in self.shared_encoder.embeddings.parameters():
            paras.requires_grad = False
        self.shared_encoder.embeddings.eval()

    def on_save_checkpoint(self, state_dict: Dict[str, Any]) -> None:
        for key in state_dict:
            if key.startswith("llm_embedding_layer"):
                del state_dict[key]

    def training_step(self, batch, batch_idx):
        losses = OrderedDict()
        for name, handle in self.handles.items():
            losses[name] = handle.train_step(self, batch, batch_idx)

        loss = 0
        for name, l in losses.items():
            loss += l

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        for name, handle in self.handles.items():
            handle.validation_step(self, batch, batch_idx)

    def on_train_epoch_end(self):
        for name, handle in self.handles.items():
            handle.on_train_epoch_end(self)

    def on_validation_epoch_end(self):
        for name, handle in self.handles.items():
            handle.on_validation_epoch_end(self)

    def train(self, is_train):
        super().train(is_train)
        self.shared_encoder.embeddings.eval()
        self.visual_encoder.eval()

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.engine.optimizer,
            [
                {"params": filter(lambda p: p.requires_grad, self.parameters())},
            ],
        )
        scheduler = instantiate(self.cfg.engine.lr_scheduler, opt)
        return {"optimizer": opt, "lr_scheduler": scheduler}


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
