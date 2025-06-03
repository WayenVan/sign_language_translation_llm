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
import numpy as np
from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration


# from modules.fsmt.modeling_fsmt import FSMTForConditionalGeneration
# from transformers import FSMTTokenizer
from modules.extended_embeddings import CustomEmbeddingLayer
from modules.q_former.q_former import (
    BertLMHeadModel,
    BertModel,
    BertConfig,
    BertOnlyMLMHead,
)
from transformers.models.bert import BertLMHeadModel as BertLMHeadModelFromHF

from .handles.vtg_handle import VTGHandle
from .handles.vtm_handle import VTMHandle
from .handles.pl_handle import PLHandle
from .handles.vtc_handle import VTCHandle

from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us


class SLTModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug

        # write only for this model
        self.vtg_flag = getattr(self.cfg, "vtg_flag", False)
        self.vtm_flag = getattr(self.cfg, "vtm_flag", False)
        self.vtc_flag = getattr(self.cfg, "vtc_flag", False)
        self.pl_flag = getattr(self.cfg, "pl_flag", False)

        self.contrastive_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self._create_llm()
        self._create_visual_layers()
        self._create_bert_shared_encoder()
        self._create_handles()

        self.freezer = instantiate(
            self.cfg.modules.freezer, visual_encoder=self.visual_encoder
        )
        # NOTE: freeze the visual encoder by freezer
        self.freezer.freeze()

    def _create_llm(self):
        if self.cfg.inference_mode or self.pl_flag:
            self.connector = instantiate(self.cfg.modules.connector)

            # mname = "WayenVan/wmt19-en-de"
            # self.llm = FSMTForConditionalGeneration.from_pretrained(
            #     mname, device_map="cpu", torch_dtype=torch.float32
            # )
            # self.llm_tokenizer = FSMTTokenizer.from_pretrained(mname)
            mname = "google/flan-t5-large"
            self.llm = T5ForConditionalGeneration.from_pretrained(
                mname, device_map="cpu", torch_dtype=torch.float32
            )
            self.llm_tokenizer = T5Tokenizer.from_pretrained(mname)

            # NOTE: freezed llm
            for paras in self.llm.parameters():
                paras.requires_grad = False
            self.llm.eval()

        else:
            self.llm = None
            self.llm_tokenizer = None
            self.connector = None

    def _create_handles(self):
        self.handles = nn.ModuleDict()

        if self.vtg_flag:
            self.handles["vtg"] = VTGHandle(self.vocab_size, self.tokenizer, self.cfg)
            self.vtg_weight = self.cfg.vtg_weight

        if self.vtm_flag:
            self.handles["vtm"] = VTMHandle(self.tokenizer, self.vocab_size, self.cfg)
            self.vtm_weight = self.cfg.vtm_weight

        if self.vtc_flag:
            self.handles["vtc"] = VTCHandle(self.hidden_size, self.cfg)
            self.vtc_weight = self.cfg.vtc_weight

        if self.pl_flag and (self.vtm_flag or self.vtg_flag or self.vtc_flag):
            raise ValueError(
                "Prompt learning is not supported with VTM or VTG. Please set prompt_learning to False."
            )

        if self.pl_flag:
            self.handles["pl"] = PLHandle(
                self,
                len(self.llm_tokenizer),
                self.cfg.pl_weight,
                self.llm_tokenizer.added_tokens_encoder["<pad>"],
            )

    def _create_visual_layers(self):
        self.visual_encoder = instantiate(self.cfg.modules.visual_encoder)
        self.visual_adapter = instantiate(self.cfg.modules.visual_adapter)

    def _create_bert_shared_encoder(self, cross_attention_freq=2):
        self.num_query_token = self.cfg.modules.num_query_token

        # setup q former configs
        self.bert_config = BertConfig.from_pretrained(
            self.cfg.modules.bert_shared_encoder_id,
        )
        self.hidden_size = self.bert_config.hidden_size

        self.bert_config.cross_attention_freq = cross_attention_freq
        self.bert_config.add_cross_attention = True
        self.bert_config.query_length = self.num_query_token
        self.bert_config.encoder_width = self.hidden_size

        self.shared_encoder = BertLMHeadModel(self.bert_config)
        params = BertLMHeadModelFromHF.from_pretrained(
            self.cfg.modules.bert_shared_encoder_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        ).state_dict()
        self.shared_encoder.load_state_dict(params, strict=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.modules.bert_shared_encoder_id,
            use_fast=True,
        )

        # NOTE: add bos and eos token
        old_vocab_size = self.tokenizer.vocab_size
        self.tokenizer.padding_side = "right"
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
            }
        )
        self.shared_encoder.resize_token_embeddings(self.tokenizer.vocab_size)
        new_vocab_size = len(self.tokenizer)
        pretrained_weights = self.shared_encoder.get_input_embeddings().weight
        padding_idx = self.tokenizer.pad_token_id

        # NOTE: createa a new embeeding layer for bert
        self.shared_encoder.bert.embeddings.word_embeddings = CustomEmbeddingLayer(
            old_vocab_size,
            2,
            self.hidden_size,
            padding_idx,
            pretrained_weights,
        )

        # update vocab size
        self.vocab_size = new_vocab_size
        assert self.vocab_size == self.bert_config.vocab_size + 2, (
            f"Vocab size {self.vocab_size} does not match bert config vocab size {self.bert_config.vocab_size}"
        )
        # updatre config and create new output layer
        self.bert_config.vocab_size = self.shared_encoder.config.vocab_size + 2
        self.shared_encoder.cls = BertOnlyMLMHead(config=self.bert_config)

        # NOTE: freeze all the embedding model in the layer, but not the new one
        for paras in self.shared_encoder.bert.embeddings.parameters():
            paras.requires_grad = False
        for paras in self.shared_encoder.bert.embeddings.word_embeddings.new_embeddings.parameters():
            paras.requires_grad = True

        # NOTE: create visual embedding for visual encoder
        self.video_query_tokens = nn.Parameter(
            torch.zeros(
                (1, self.num_query_token, self.hidden_size),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self.video_query_tokens.data.normal_(
            mean=0.0, std=self.shared_encoder.config.initializer_range
        )

    def forward_visual(self, video, video_length):
        """
        Forward pass for visual features.
        :param video: (batch_size, seq_len, 3, h, w)
        :param video_length: (batch_size,)
        :return: visual embeddings
        """
        visual_outputs = self.visual_encoder(video, video_length)
        v_length = visual_outputs.video_length
        visual_embeddings, v_length = self.visual_adapter(
            visual_outputs.hidden_state, v_length
        )
        return visual_embeddings, v_length

    def training_step(self, batch, batch_idx):
        # forward visual features to avoid duplicated memory computation
        video = batch["video"].to(self.device)
        video_length = batch["video_length"].to(self.device)
        visual_embeddings, v_length = self.forward_visual(video, video_length)

        losses = OrderedDict()
        for name, handle in self.handles.items():
            losses[name] = handle.train_step(
                self, batch, batch_idx, visual_embeddings, v_length
            )
        loss = 0
        for name, l in losses.items():
            loss += l

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # forward visual features to avoid duplicated memory computation
        video = batch["video"].to(self.device)
        video_length = batch["video_length"].to(self.device)
        visual_embeddings, v_length = self.forward_visual(video, video_length)

        for name, handle in self.handles.items():
            handle.validation_step(self, batch, batch_idx, visual_embeddings, v_length)

    def on_train_epoch_end(self):
        for name, handle in self.handles.items():
            handle.on_train_epoch_end(self)

    def on_validation_epoch_end(self):
        for name, handle in self.handles.items():
            handle.on_validation_epoch_end(self)

    def train(self, is_train):
        super().train(is_train)
        self.shared_encoder.bert.embeddings.eval()
        self.visual_encoder.eval()

        for name, handle in self.handles.items():
            handle.train_handle(self, is_train)

        # NOTE: delegate the train to the freezer
        self.freezer.train(is_train)

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.engine.optimizer,
            [
                {"params": filter(lambda p: p.requires_grad, self.parameters())},
            ],
        )
        scheduler = instantiate(self.cfg.engine.lr_scheduler, opt)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name in list(checkpoint.keys()):
            if name.startswith("llm"):
                del checkpoint[name]
        return checkpoint

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

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor = None,
        video_length: torch.Tensor = None,
        video_embeddings: Optional[torch.Tensor] = None,
        max_length: Optional[int] = 30,
    ):
        """
        @param video: (batch_size, seq_len, 3, h, w)
        """
        if video is None and video_embeddings is None:
            raise ValueError("Either video or video_embeddings must be provided.")

        if video is not None and video_embeddings is not None:
            raise ValueError(
                "Only one of video or video_embeddings should be provided."
            )

        if video is not None:
            B, T, _, _, _ = video.shape
            device = video.device
            video_embeddings, video_length = self.forward_visual(video, video_length)
        else:
            B, T, _ = video_embeddings.shape
            device = video_embeddings.device

        # Create video mask
        video_attention_mask = self.length_to_mask(
            video_length, max_length=video_embeddings.shape[1]
        )
        # video query
        video_query_tokens = self.video_query_tokens.expand(B, -1, -1)

        with torch.no_grad():
            # attention mask for the shared encoder
            attn_mask = torch.ones(
                B, self.num_query_token + 1, device=device, dtype=torch.long
            )

            # Initialize generation state
            output = []
            unfinished = torch.ones(B, dtype=torch.bool, device=device)
            input = torch.full(
                (B, 1), self.tokenizer.bos_token_id, device=device, dtype=torch.long
            )
            for _ in range(max_length):
                if not unfinished.any():
                    break

                # Forward pass
                logits = self.shared_encoder(
                    input_ids=input,
                    query_embeds=video_query_tokens,
                    attention_mask=attn_mask,
                    encoder_hidden_states=video_embeddings,
                    encoder_attention_mask=video_attention_mask,
                ).logits  # [B, L, C]
                logits = logits[:, -1, :]  # Get logits for the last token
                # [B, C]

                next_tokens = torch.argmax(logits, dim=-1)  # Greedy decoding
                # [B ]

                # Update unfinished sequences
                unfinished = unfinished & (
                    next_tokens.squeeze() != self.tokenizer.eos_token_id
                )

                # update output
                output.append(next_tokens.unsqueeze(-1))  # [B, 1]

                # Prepare next inputs
                input = torch.cat((input, next_tokens.unsqueeze(-1)), dim=1)
                attn_mask = torch.cat(
                    (attn_mask, unfinished.unsqueeze(-1).long()), dim=1
                )

            # [B L]
            return (
                torch.cat(output, dim=1)
                if len(output) > 0
                else torch.empty(B, 0, device=device)
            )


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
