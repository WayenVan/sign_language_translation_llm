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

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics import Accuracy, BLEUScore
import copy

# logger = logging.getLogger(__name__)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class SLTModelForT5FineTune(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg: DictConfig = cfg
        self.num_global_tokens = cfg.modules.num_global_tokens
        self._init_t5_model()

        self.text_global_tokens = nn.Parameter(
            torch.randn(1, self.num_global_tokens, self.d_model),
            requires_grad=True,
        )

        self.vision_global_tokens = nn.Parameter(
            torch.randn(1, self.num_global_tokens, self.d_model),
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
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        keys_in_ckpt = set(state_dict.keys())
        for name, p in self.named_parameters():
            if name.startswith("t5."):
                assert name in keys_in_ckpt, (
                    f"Parameter {name} not found in the checkpoint"
                )

        keys = self.load_state_dict(ckpt["state_dict"], strict=False)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            # NOTE: print information
            for key in keys.missing_keys:
                if not key.startswith("llm.") and not key.startswith("connector."):
                    print(f"Missing key {key} in the state dict")
            for key in keys.unexpected_keys:
                print(f"Unexpected key {key} in the state dict")

        self.t5.to(torch.bfloat16)  # Use bfloat16 for better performance on TPUs

    def _init_visual_modules(self):
        self.visual_backbone = instantiate(self.cfg.modules.backbone)
        self.visual_adapter = instantiate(self.cfg.modules.visual_adapter)
        self.connector = build_mlp(
            self.cfg.modules.connector_depth, self.d_model, self.d_model
        )

        self.visual_encoder_cfg = copy.deepcopy(self.t5_config)
        self.visual_encoder_cfg.is_decoder = False
        self.visual_encoder_cfg.use_cache = False
        self.visual_encoder_cfg.is_encoder_decoder = False
        self.visual_encoder_cfg.num_layers = (
            self.t5_config.num_layers * self.cfg.modules.visual_encoder_layer_scale
        )
        self.visual_encoder = T5Stack(self.visual_encoder_cfg)

        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        self.visual_backbone.eval()

    def _init_t5_model(self):
        mname = self.cfg.modules.t5_model_name
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            mname,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on TPUs
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(mname)
        self.t5_config = T5Config.from_pretrained(mname)
        self.d_model = self.t5_config.d_model

        self.eos_token = "</s>"
        self.bos_token = "<pad>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.bos_token)

        # freeze the t5 model
        for param in self.t5.parameters():
            param.requires_grad = False
        self.t5.eval()

    def get_eos_embedding(self):
        """
        Get the embedding for the end-of-sequence token.
        """
        eos_token_id = self.eos_token_id
        eos_embedding = self.t5.get_input_embeddings()(
            torch.tensor([eos_token_id])
        ).unsqueeze(0)  # [1, 1, D]
        return eos_embedding

    def get_bos_embedding(self):
        """
        Get the embedding for the beginning-of-sequence token.
        """
        bos_token_id = self.bos_token_id
        bos_embedding = self.t5.get_input_embeddings()(
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
        G = self.num_global_tokens

        video, video_length = self.visual_backbone(
            video, video_length
        )  # [B, T, H, W, C]
        video_feats, video_length = self.visual_adapter(
            video, video_length
        )  # [B, T, D], [B, T]

        global_embeddings = self.vision_global_tokens.expand(B, G, -1)

        inputs_embeds = torch.cat([global_embeddings, video_feats], dim=1)

        video_length = video_length + G  # Add global tokens to length

        attention_mask = self.length_to_mask(video_length)

        encoder_outputs = self.visual_encoder(
            inputs_embeds=inputs_embeds,  # [B, T, D]
            attention_mask=attention_mask,  # [B, T]
        )
        visual_global_embeddings = encoder_outputs.last_hidden_state[
            :, :G, :
        ]  # [B, G, D]
        visual_connected_embeddings = self.connector(visual_global_embeddings)
        return visual_connected_embeddings, visual_global_embeddings

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
        """
        Forward pass through the T5 model with global tokens.
        """

        B, _ = input_ids.shape
        G = self.num_global_tokens

        text_embeddings = self.t5.get_input_embeddings()(input_ids)  # [B, L, D]
        global_embeddings = self.text_global_tokens.expand(B, -1, -1)  # [B, G, D]

        text_embeddings = torch.cat([global_embeddings, text_embeddings], dim=1)

        attention_mask = F.pad(attention_mask, (G, 0), value=1).long()  # [B, G + L]

        encoder_outputs = self.t5.encoder(
            inputs_embeds=text_embeddings,
            # attention_mask=attention_mask,
        )
        global_outputs = encoder_outputs.last_hidden_state[:, :G, :]  # [B, G, D]
        return global_outputs

    def decoder_forward(
        self,
        global_embeddings,  # [B, G, D]
        decoder_input_ids: Tensor,  # [B, L]
    ):
        """
        Forward pass through the T5 decoder.
        """
        decoder_input_embeddings = self.t5.get_input_embeddings()(
            decoder_input_ids
        )  # [B, L, D]
        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_input_embeddings,
            encoder_hidden_states=global_embeddings,
        )
        logits = self.t5.lm_head(decoder_outputs.last_hidden_state)  # [B, L, C]
        return (
            logits,
            decoder_outputs.last_hidden_state,
        )  # [B, L, C], [B, L, D]

    def tokenize_texts(self, texts: List[str]):
        tokenized = self.tokenizer(
            texts,
            text_target=texts,
            padding=True,
            # truncation=False,
            # max_length=self.t5_config.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)  # [B, L]
        attention_mask = tokenized.attention_mask.to(self.device)  # [B, L]

        label_ids = tokenized.labels.to(self.device)  # [B, L]
        decoder_input_ids = torch.cat(
            [
                torch.full(
                    (label_ids.shape[0], 1),
                    self.bos_token_id,
                    dtype=torch.long,
                ).to(self.device),  # [B, 1]
                label_ids,
            ],
            dim=1,
        )
        label_ids[
            ~attention_mask.bool()
        ] = -100  # Set padding tokens to -100 for loss calculation
        return input_ids, attention_mask, decoder_input_ids, label_ids

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
        visual_global_embeddings: Tensor,  # [B, G, D]
        text_global_embeddings: Tensor,  # [B, G, D]
    ):
        """
        Calculate the loss for the visual-text alignment.
        This is a placeholder method and should be implemented based on the specific task.
        """
        # first try simple mse
        text_global_embeddings = text_global_embeddings.detach()
        text_global_embeddings = F.normalize(text_global_embeddings, dim=-1)
        visual_global_embeddings = F.normalize(visual_global_embeddings, dim=-1)

        mse = (
            F.mse_loss(
                visual_global_embeddings, text_global_embeddings, reduction="none"
            )
            .sum(dim=-1)
            .mean()
        )
        return mse

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        ids, video, video_length, text = self.dispatch_batch(batch, self.device)

        input_ids, attention_mask, decoder_input_ids, labels = self.tokenize_texts(text)

        text_global_embeddings = self.text_encoder_forward(input_ids, attention_mask)

        visual_global_embeddings, visual_connected_embeddings = (
            self.visual_encoder_forward(video, video_length)
        )
        # Calculate visual-text loss
        visual_text_loss = (
            self.visual_text_loss(visual_global_embeddings, text_global_embeddings)
            * self.cfg.visual_text_alignment_weight
        )
        self.log(
            "train_visual_text_loss",
            visual_text_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        logits, _ = self.decoder_forward(visual_connected_embeddings, decoder_input_ids)
        # Calculate loss
        LABEL_LENGTH = labels.shape[1]
        avialiable_logits = logits[:, :-1, :]

        generate_loss = F.cross_entropy(
            rearrange(avialiable_logits, "b l c -> (b l) c"),
            rearrange(labels, "b l -> (b l)"),
            ignore_index=-100,  # T5 uses -100
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
        id, video, video_length, text = self.dispatch_batch(batch, self.device)
        input_ids, attention_mask, decoder_input_ids, labels = self.tokenize_texts(text)
        _, visual_connected_embeddings = self.visual_encoder_forward(
            video, video_length
        )

        output = self.t5.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=visual_connected_embeddings,
                hidden_states=None,
                attentions=None,
            ),
            max_length=50,
        )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        reference = [[t] for t in text]
        self.bleu.update(decoded_output, reference)
        self.blue4.update(decoded_output, reference)
        return output

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
