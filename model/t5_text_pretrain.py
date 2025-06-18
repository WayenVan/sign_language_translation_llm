from einops import rearrange
from numpy import dtype
import torch
from torch import nn, Tensor
from lightning import LightningModule

from omegaconf import OmegaConf, DictConfig

from hydra.utils import instantiate

from typing import List
from transformers import get_scheduler
from torch.optim import Optimizer
from torch.nn import functional as F

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.modeling_outputs import BaseModelOutput
from torchmetrics import Accuracy, BLEUScore


class ModelForT5TextPretrain(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg: DictConfig = cfg
        self.num_global_tokens = cfg.modules.num_global_tokens
        self._init_t5_model()

        self.text_global_tokens = nn.Parameter(
            torch.randn(1, self.num_global_tokens, self.d_model),
            requires_grad=True,
        )
        self.train_accu = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=0,  # padding index
        )
        self.bleu = BLEUScore(n_gram=1, smooth=True)
        self.blue4 = BLEUScore(n_gram=4, smooth=True)

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

        # unfreeze the encoder
        for name, param in self.t5.encoder.named_parameters():
            param.requires_grad = True

        for param in self.t5.shared.parameters():
            param.requires_grad = False

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
            max_length=self.t5_config.max_length,
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

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        ids, video, video_length, text = self.dispatch_batch(batch, self.device)

        input_ids, attention_mask, decoder_input_ids, labels = self.tokenize_texts(text)
        global_embeddings = self.text_encoder_forward(input_ids, attention_mask)

        logits, _ = self.decoder_forward(global_embeddings, decoder_input_ids)
        # Calculate loss
        LABEL_LENGTH = labels.shape[1]
        avialiable_logits = logits[:, :-1, :]

        loss = F.cross_entropy(
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
        self.log("train_text_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        id, video, video_length, text = self.dispatch_batch(batch, self.device)
        input_ids, attention_mask, decoder_input_ids, labels = self.tokenize_texts(text)
        global_embeddings = self.text_encoder_forward(input_ids, attention_mask)

        output = self.t5.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=global_embeddings,
                hidden_states=None,
                attentions=None,
            ),
            max_length=50,
        )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        reference = [[t] for t in text]
        self.bleu.update(decoded_output, reference)
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
