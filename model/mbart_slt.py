import logging


from einops import rearrange
import torch
from torch import nn, Tensor
from lightning import LightningModule
import os


from omegaconf import OmegaConf, DictConfig

from hydra.utils import instantiate

from typing import List
from transformers import get_scheduler
from torch.optim import Optimizer
from torch.nn import functional as F

# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
# from transformers.models.t5.configuration_t5 import T5Config
# from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.mbart50.tokenization_mbart50_fast import MBart50TokenizerFast
from transformers.models.mbart.modeling_mbart import (
    MBartEncoder,
    MBartForConditionalGeneration,
)
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics import Accuracy, BLEUScore
from typing import Any
import copy

# logger = logging.getLogger(__name__)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class MBartSLTModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg: DictConfig = cfg
        self.lang = "de_DE"  # Set source language to German
        self._init_mbart_model()

        self.visual_src_lang_token = nn.Parameter(
            torch.randn(1, 1, self.d_model),
            requires_grad=True,
        )

        self.train_accu = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=0,  # padding index
        )

        self._init_visual_modules()

        self.bleu = BLEUScore(n_gram=1, smooth=True)
        self.blue4 = BLEUScore(n_gram=4, smooth=True)

    def load_from_pretrained(self, path: str):
        """
        Load the model from a pretrained T5TextPretrain model.
        """
        # ckpt = torch.load(path, map_location="cpu")
        # state_dict = ckpt["state_dict"]
        # keys_in_ckpt = set(state_dict.keys())
        # for name, p in self.named_parameters():
        #     if name.startswith("t5."):
        #         assert name in keys_in_ckpt, (
        #             f"Parameter {name} not found in the checkpoint"
        #         )
        #
        # keys = self.load_state_dict(ckpt["state_dict"], strict=False)
        #
        # if os.environ.get("LOCAL_RANK", "0") == "0":
        #     # NOTE: print information
        #     for key in keys.missing_keys:
        #         if not key.startswith("llm.") and not key.startswith("connector."):
        #             print(f"Missing key {key} in the state dict")
        #     for key in keys.unexpected_keys:
        #         print(f"Unexpected key {key} in the state dict")
        #
        # self.t5.to(torch.bfloat16)  # Use bfloat16 for better performance on TPUs

    def _init_visual_modules(self):
        self.visual_backbone = instantiate(self.cfg.modules.backbone)
        self.visual_adapter = instantiate(self.cfg.modules.visual_adapter)
        self.connector = build_mlp(
            self.cfg.modules.connector_depth, self.d_model, self.d_model
        )

        self.visual_encoder_cfg = copy.deepcopy(self.mbart_config)
        self.visual_encoder_cfg.num_layers = (
            self.mbart_config.encoder_layers
            * self.cfg.modules.visual_encoder_layer_scale
        )
        self.visual_encoder = MBartEncoder(self.visual_encoder_cfg)

        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        self.visual_backbone.eval()

    def _init_mbart_model(self):
        mname = "facebook/mbart-large-50-many-to-many-mmt"
        self.mbart = MBartForConditionalGeneration.from_pretrained(
            mname,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on TPUs
        )
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            mname,
        )
        self.tokenizer.src_lang = self.lang
        self.mbart_config = MBartConfig.from_pretrained(mname)
        self.d_model = self.mbart_config.d_model

        self.eos_token = "</s>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

        # freeze the mbart model except the decoder
        for param in self.mbart.parameters():
            param.requires_grad = False
        # for param in self.mbart.base_model.decoder.parameters():
        #     param.requires_grad = True
        # for param in self.mbart.shared.parameters():
        #     param.requires_grad = False

    def get_eos_embedding(self):
        """
        Get the embedding for the end-of-sequence token.
        """
        eos_token_id = self.eos_token_id
        eos_embedding = self.mbart.get_input_embeddings()(
            torch.tensor([eos_token_id])
        ).unsqueeze(0)  # [1, 1, D]
        return eos_embedding

    def get_bos_embedding(self):
        """
        Get the embedding for the beginning-of-sequence token.
        """
        bos_token_id = self.bos_token_id
        bos_embedding = self.mbart.get_input_embeddings()(
            torch.tensor([bos_token_id])
        ).unsqueeze(0)  # [1, 1, D]
        return bos_embedding

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        """
        train_acc = self.train_accu.compute()
        self.log("train_generate_accu", train_acc, prog_bar=True, sync_dist=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        bleu = self.bleu.compute()
        self.log("val_generate_bleu", bleu, prog_bar=True, sync_dist=True)
        self.bleu.reset()

        bleu4 = self.blue4.compute()
        self.log("val_generate_bleu4", bleu4, prog_bar=True, sync_dist=True)
        self.blue4.reset()

    def visual_encoder_forward(self, video: Tensor, video_length: Tensor):
        """
        Forward pass through the visual encoder.
        """
        B, T, C, H, W = video.shape

        video, video_length = self.visual_backbone(
            video, video_length
        )  # [B, T, H, W, C]
        video_feats, video_length = self.visual_adapter(
            video, video_length
        )  # [B, T, D], [B, T]

        visual_src_lang_token = self.visual_src_lang_token.expand(B, 1, -1)

        inputs_embeds = torch.cat(
            [visual_src_lang_token, video_feats], dim=1
        )  # [B, 1 + T, D]

        video_length = video_length + 1

        attention_mask = self.length_to_mask(video_length)

        encoder_outputs = self.visual_encoder(
            inputs_embeds=inputs_embeds,  # [B, T, D]
            attention_mask=attention_mask,  # [B, T]
        )

        visual_last_hidden_states = encoder_outputs.last_hidden_state  # [B, T, D]
        visual_lang_feats = visual_last_hidden_states[:, 0, :]  # [B, D] (global token)

        visual_feats = self.connector(visual_last_hidden_states)  # [B, T, D]
        return (
            visual_feats,
            visual_lang_feats,
            visual_last_hidden_states,  # [B, T, D]
            attention_mask,
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

    def text_encoder_forward(self, input_ids: Tensor, attention_mask: Tensor):
        B, _ = input_ids.shape

        encoder_outputs = self.mbart.base_model.encoder(
            input_ids=input_ids,  # [B, L, D]
            attention_mask=attention_mask,  # [B, L]
        )
        text_last_hidden_states = encoder_outputs.last_hidden_state  # [B, L, D]
        text_lang_feats = text_last_hidden_states[:, 0, :]  # [B, D] (global token)
        return text_last_hidden_states, text_lang_feats  # [B, L, D], [B, D]

    def decoder_forward(
        self,
        encoder_hidden_states,  # [B, G, D]
        decoder_input_ids: Tensor,  # [B, L]
        encoder_attn_mask: Tensor
        | None = None,  # [B, L] (optional, not used in this case)
        decoder_attn_mask: Tensor
        | None = None,  # [B, L] (optional, not used in this case)
    ):
        """
        Forward pass through the T5 decoder.
        """
        decoder_input_embeddings = self.mbart.get_input_embeddings()(
            decoder_input_ids
        )  # [B, L, D]
        decoder_outputs = self.mbart.base_model.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attn_mask,  # [B, T] (optional, not used in this case)
        )
        logits = self.mbart.lm_head(decoder_outputs.last_hidden_state)  # [B, L, C]
        return (
            logits,
            decoder_outputs.last_hidden_state,
        )  # [B, L, C], [B, L, D]

    def tokenize_texts(self, texts: List[str]):
        tokenized = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)  # [B, L]
        attention_mask = tokenized.attention_mask.to(self.device)  # [B, L]

        label_ids = input_ids.clone()  # [B, L]

        decoder_input_ids = torch.cat(
            [
                torch.full(
                    (label_ids.shape[0], 1),
                    self.eos_token_id,
                    dtype=torch.long,
                ).to(self.device),  # [B, 1]
                label_ids,
            ],
            dim=1,
        )
        decoder_attention_mask = torch.cat(
            [
                torch.ones(
                    (label_ids.shape[0], 1), dtype=torch.long, device=self.device
                ),  # [B, 1]
                attention_mask,
            ],
            dim=1,
        )

        label_ids[
            ~attention_mask.bool()
        ] = -100  # Set padding tokens to -100 for loss calculation

        return (
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            label_ids,
        )

    def dispatch_batch(
        self, batch, device
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[str]]:
        ids: list[str] = batch["id"]
        video: torch.Tensor = batch["video"].to(device)
        video_length: torch.Tensor = batch["video_length"].to(device)
        text: list[str] = batch["text"]
        return ids, video, video_length, text

    def visual_text_loss(
        self,
        visual_embedding: Tensor,  # [B,  D]
        text_embedding: Tensor,  # [B, D]
    ):
        """
        Calculate the loss for the visual-text alignment.
        This is a placeholder method and should be implemented based on the specific task.
        """
        # first try simple mse
        text_embedding = text_embedding.detach()

        text_embedding = F.normalize(text_embedding, dim=-1)
        visual_embedding = F.normalize(visual_embedding, dim=-1)

        mse = (
            F.mse_loss(visual_embedding, text_embedding, reduction="none")
            .sum(dim=-1)
            .mean()
        )
        return mse

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        ids, video, video_length, text = self.dispatch_batch(batch, self.device)

        # input_ids, attention_mask, decoder_input_ids, labels = self.tokenize_texts(text)
        input_ids, attention_mask, decoder_input_ids, decoder_attn_mask, labels = (
            self.tokenize_texts(text)
        )  # [B, L], [B, L], [B, L], [B, L], [B, L]

        text_feats, text_lang_feats = self.text_encoder_forward(
            input_ids, attention_mask
        )

        (
            visual_feats,
            visual_lang_feats,
            visual_last_hidden_states,  # [B, T, D]
            visual_attn_mask,
        ) = self.visual_encoder_forward(video, video_length)

        # Calculate visual-text loss
        visual_text_loss = (
            self.visual_text_loss(visual_lang_feats, text_lang_feats)
            * self.cfg.visual_text_alignment_weight
        )
        self.log(
            "train_visual_text_loss",
            visual_text_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # logits, _ = self.decoder_forward(visaul_connected_embeddings, decoder_input_ids)
        logits, _ = self.decoder_forward(
            visual_feats, decoder_input_ids, visual_attn_mask, decoder_attn_mask
        )
        # Calculate loss
        LABEL_LENGTH = labels.shape[1]
        avialiable_logits = logits[:, :-1, :]

        generate_loss = F.cross_entropy(
            rearrange(avialiable_logits, "b l c -> (b l) c"),
            rearrange(labels, "b l -> (b l)"),
            ignore_index=-100,
        )

        # Update accuracy
        self.train_accu.update(
            rearrange(avialiable_logits, "b l c -> (b l) c")[
                :, : self.tokenizer.vocab_size
            ],
            rearrange(labels, "b l -> (b l)"),
        )
        self.log(
            "train_generate_loss",
            generate_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        total_loss = visual_text_loss + generate_loss
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        ids, video, video_length, text = self.dispatch_batch(batch, self.device)

        # tokenize the text
        input_ids, attention_mask, decoder_input_ids, decoder_attn_mask, labels = (
            self.tokenize_texts(text)
        )  # [B, L]

        # visual forward
        (
            visual_feats,
            visual_lang_feats,
            visual_last_hidden_states,  # [B, T, D]
            visual_attn_mask,
        ) = self.visual_encoder_forward(video, video_length)

        output = self.mbart.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=visual_feats,
                hidden_states=None,
                attentions=None,
            ),
            attention_mask=visual_attn_mask,  # [B, T]
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang],
            max_length=50,
        )

        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        reference = [[t] for t in text]
        self.bleu.update(decoded_output, reference)
        self.blue4.update(decoded_output, reference)
        return output

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.mbart.to(torch.bfloat16)  # Use bfloat16 for better performance on TPUs

    @staticmethod
    def get_lr_schuduler(cfg, optimizer: Optimizer):
        lr_config = cfg.engine.lr_scheduler
        if lr_config.type == "native" or lr_config.type == "timm":
            return instantiate(lr_config.instance, optimizer=optimizer)
        elif lr_config.type == "transformers":
            # scheduler = instantiate(self.cfg.engine.lr_scheduler, opt)
            kwarges = cfg.engine.lr_scheduler.kwargs
            if kwarges is None:
                kwarges = {}
            else:
                kwarges = OmegaConf.to_container(kwarges, resolve=True)

            scheduler = get_scheduler(
                cfg.engine.lr_scheduler.name,
                optimizer=optimizer,
                num_warmup_steps=cfg.engine.lr_scheduler.warmup_steps,
                num_training_steps=cfg.engine.lr_scheduler.training_steps,
                scheduler_specific_kwargs=kwarges,
            )
            return scheduler
        else:
            raise ValueError(
                f"Unsupported lr scheduler type: {cfg.engine.lr_scheduler.type}"
            )

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.engine.optimizer,
            [
                {"params": filter(lambda p: p.requires_grad, self.parameters())},
            ],
        )
        scheduler = self.get_lr_schuduler(self.cfg, opt)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'epoch'
            },
        }
