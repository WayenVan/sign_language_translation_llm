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
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration

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

        self._create_llm()
        self._create_layers()
        self._create_bert_shared_encoder()
        self._create_handles()

    def _create_llm(self):
        self.connector = instantiate(self.cfg.modules.connector)
        if self.cfg.inference_mode or self.pl_flag:
            self.llm = Gemma3ForCausalLM.from_pretrained(
                "google/gemma-3-1b-it",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-3-1b-it",
                use_fast=True,
            )
            self.llm_tokenizer.padding_side = "right"

            # NOTE: freezed llm
            for paras in self.llm.parameters():
                paras.requires_grad = False

        else:
            self.llm = None
            self.llm_tokenizer = None

    def _create_handles(self):
        self.handles = nn.ModuleDict()

        if self.vtg_flag:
            self.handles["vtg"] = VTGHandle(self.vocab_size, self.cfg.vtg_weight)
            self.vtg_weight = self.cfg.vtg_weight

        if self.vtm_flag:
            self.handles["vtm"] = VTMHandle(
                self.vocab_size,
                self.cfg.vtm_mask_ratio,
            )
            self.vtm_weight = self.cfg.vtm_weight

        if self.vtc_flag:
            self.handles["vtc"] = VTCHandle(
                self.cfg.vtc_weight,
            )
            self.vtc_weight = self.cfg.vtc_weight

        if self.pl_flag and (self.vtm_flag or self.vtg_flag or self.vtc_flag):
            raise ValueError(
                "Prompt learning is not supported with VTM or VTG. Please set prompt_learning to False."
            )

        if self.pl_flag:
            self.handles["pl"] = PLHandle(
                self,
                self.llm_tokenizer.vocab_size,
                self.cfg.pl_weight,
                self.llm_tokenizer.pad_token_type_id,
            )

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
            self.vocab_size,  # WARN: be careful with the vocab size and len(tokenizer)
        )

        # NOTE: freeze all the embedding model in the layer
        for paras in self.shared_encoder.embeddings.parameters():
            paras.requires_grad = False
        self.shared_encoder.embeddings.eval()

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

        for name, handle in self.handles.items():
            handle.train_handle(self, is_train)

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
            if name.startswith("visual_encoder"):
                del checkpoint[name]
        return checkpoint

    def generate(
        self,
        video: torch.Tensor,
        video_length: torch.Tensor,
        max_length: Optional[int] = 30,
    ):
        """
        @param video: (batch_size, seq_len, 3, h, w)
        """
        B, T, C, H, W = video.shape
        device = video.device

        # Create video mask
        video_mask = torch.arange(T, device=device).expand(
            B, T
        ) < video_length.unsqueeze(1)
        video_mask = video_mask.long()

        with torch.no_grad():
            # Process visual inputs
            visual_outputs = self.visual_encoder(video, video_length)
            visual_embeddings = self.visual_adapter(visual_outputs.hidden_state)

            # Prepare initial inputs with CLS token
            cls_embeddings = self.shared_encoder.get_input_embeddings()(
                torch.full((B, 1), self.tokenizer.cls_token_id, device=device)
            )
            inputs = torch.cat((visual_embeddings, cls_embeddings), dim=1)
            attn_mask = torch.cat(
                (video_mask, torch.ones((B, 1), device=device)), dim=1
            )

            # Initialize generation state
            output = []
            unfinished = torch.ones(B, dtype=torch.bool, device=device)

            for _ in range(max_length):
                if not unfinished.any():
                    break

                # Forward pass
                out_features = self.shared_encoder(
                    inputs_embeds=inputs,
                    attention_mask=attn_mask,
                )

                # Get next token predictions
                logits = self.shared_encoder_header(
                    out_features.last_hidden_state[:, -1:]
                )
                next_tokens = torch.argmax(logits, dim=-1)  # Greedy decoding

                # Update unfinished sequences
                unfinished = unfinished & (
                    next_tokens.squeeze() != self.tokenizer.sep_token_id
                )
                output.append(next_tokens)

                # Prepare next inputs
                inputs = torch.cat(
                    (inputs, self.shared_encoder.get_input_embeddings()(next_tokens)),
                    dim=1,
                )
                attn_mask = torch.cat(
                    (attn_mask, unfinished.unsqueeze(-1).long()), dim=1
                )

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
