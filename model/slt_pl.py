import torch
from lightning import LightningModule
from transformers import AutoTokenizer, AutoConfig
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Dict, Any
import logging
from typing import Optional, List
from einops import rearrange
import numpy as np
from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from torchmetrics import Accuracy
from torchmetrics.text import BLEUScore


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


from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us


class SLTModelForLLMFineTune(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug

        self._create_llm()
        self._create_visual_layers()
        self._create_bert_shared_encoder()

        self.train_accu = Accuracy(
            task="multiclass",
            num_classes=self.llm_vocab_size,
            # ignore_index=llm_padding_idx,  # padding index
            ignore_index=-100,
        )
        self.bleu = BLEUScore(n_gram=1, smooth=True)

        # NOTE: freeze the visual encoder
        for paras in self.visual_encoder.parameters():
            paras.requires_grad = False
        self.visual_encoder.eval()

        # NOTE: freezed llm
        for paras in self.llm.parameters():
            paras.requires_grad = False
        self.llm.eval()

        # NOTE: freeze adapter, and shared encoder
        for param in self.visual_adapter.parameters():
            param.requires_grad = False
        self.visual_adapter.eval()
        # module.shared_encoder.eval()

    def _create_llm(self):
        self.connector = instantiate(self.cfg.modules.connector)

        # mname = "WayenVan/wmt19-en-de"
        # self.llm = FSMTForConditionalGeneration.from_pretrained(
        #     mname, device_map="cpu", torch_dtype=torch.float32
        # )
        # self.llm_tokenizer = FSMTTokenizer.from_pretrained(mname)
        # mname = "google/flan-t5-large"
        # self.llm = T5ForConditionalGeneration.from_pretrained(
        #     mname, device_map="cpu", torch_dtype=torch.float32
        # )
        # self.llm_tokenizer = T5Tokenizer.from_pretrained(mname)
        mname = "google/gemma-3-4b-it"
        self.llm = Gemma3ForCausalLM.from_pretrained(
            mname, device_map="cpu", torch_dtype=torch.bfloat16
        )
        self.llm_tokenizer = GemmaTokenizerFast.from_pretrained(mname)
        self.llm_config = AutoConfig.from_pretrained(mname)
        self.llm_tokenizer.padding_side = "right"
        self.llm_hidden_size = self.llm_config.text_config.hidden_size
        self.llm_vocab_size = self.llm_tokenizer.vocab_size

        # add sepcial tokens, begain of video, start of translation
        self.llm_bov_token = nn.Parameter(
            torch.randn(1, 1, self.llm_hidden_size), requires_grad=True
        )
        self.llm_soft_token = nn.Parameter(
            torch.randn(1, 20, self.llm_hidden_size), requires_grad=True
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

    def dispatch_batch(self, batch, device):
        ids = batch["ids"]
        video = batch["video"].to(device)
        video_length = batch["video_length"].to(device)
        text = batch["text"]

        return ids, video, video_length, text

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

    @staticmethod
    def tokenize(text: List[str], tokenizer, device):
        """
        Tokenize the text using the tokenizer.
        """
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        input_outputs = tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        )

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = True

        label_outputs = tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        )

        label_outputs["input_ids"] = label_outputs["input_ids"].masked_fill(
            label_outputs["input_ids"] == tokenizer.pad_token_id,
            -100,
        )

        assert (
            input_outputs["input_ids"].shape[1]
            == label_outputs["input_ids"].shape[1] - 1
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
        visual_features,  # [b t c]
        v_length,  # [b]
        output_attentions=False,
    ):
        B, T, C = visual_features.shape
        NUM_QUERIES = self.num_query_token

        # video query
        video_query_tokens = self.video_query_tokens.expand(B, -1, -1)

        # create padding mask for the cross atttention with video
        cross_attention_mask = self.length_to_mask(v_length, max_length=T)

        # q-former forward
        bert_output = self.shared_encoder(
            attention_mask=torch.ones(
                B, NUM_QUERIES, NUM_QUERIES
            )  # WARN: need to pass the attentiom mask to avoid q_former ask shape of input_ids
            .to(self.device)
            .long(),
            query_embeds=video_query_tokens,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=cross_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        visual_features = bert_output.hidden_states[-1]
        visual_features = self.connector(visual_features)

        assert visual_features.shape[1] == NUM_QUERIES
        return visual_features

    def _prepare_llm_prompt(self, visual_features):
        """
        Prepare the input for the LLM encoder.
        [<bos> visual_features <soft_token> <bos> prompt + .....]
        visual_features: [B, NUM_QUERIES, C]
        """
        B, NUM_QUERIES, C = visual_features.shape

        bos_embed = self.llm.get_input_embeddings()(
            torch.tensor([[self.llm_tokenizer.bos_token_id]], device=self.device).long()
        )  # [1, 1, C]

        prompt = "translate to german: "
        prompt_ids = self.llm_tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)  # [1, L]
        prompt_embedding = self.llm.get_input_embeddings()(prompt_ids)  # [1, L, C]

        llm_prompt = torch.cat(
            [
                self.llm_bov_token.expand(B, -1, C),  # [B, 1, C]
                visual_features,
                self.llm_soft_token.expand(B, -1, C),  # [B, SOFT_LENGTH, C]
                bos_embed.expand(B, -1, -1),  # [B, 1, C]
                prompt_embedding.expand(B, -1, C),  # [B, L, C]
            ],
            dim=1,  # [b, 1+num_queries+SOFT_LENGTH+1+L, C]
        )
        return llm_prompt

    def _forward_llm(self, visual_features, text_ids, text_attention_mask, labels=None):
        B, _, _ = visual_features.shape

        llm_prompt = self._prepare_llm_prompt(visual_features)
        # [B, prompt_length+num_queries, C]

        text_embedding = self.llm.get_input_embeddings()(text_ids)  # [B, L, C]

        _, TEXT_LENGTH, _ = text_embedding.shape

        attention_mask = torch.cat(
            [
                torch.ones(B, llm_prompt.shape[1], device=llm_prompt.device),
                text_attention_mask,
            ],
            dim=1,
        )

        llm_outputs = self.llm(
            inputs_embeds=torch.cat(
                [
                    llm_prompt,  # [B, total_prompt_length, C]
                    text_embedding,  # [B, L, C]
                ],
                dim=1,  # [B, total_prompt_length+L, C]
            ),
            attention_mask=attention_mask,
            labels=labels,  # [B, L+1]
            use_cache=False,
            logits_to_keep=TEXT_LENGTH + 1,
        )
        return llm_outputs

    def training_step(self, batch, batch_idx):
        # forward visual features to avoid duplicated memory computation
        ids, video, video_length, text = self.dispatch_batch(batch, self.device)
        visual_embeddings, v_length = self.forward_visual(video, video_length)
        text_ids, labels, text_mask = self.tokenize(
            text, self.llm_tokenizer, self.device
        )
        visual_features = self._forward_q_former(visual_embeddings, v_length)
        llm_output = self._forward_llm(visual_features, text_ids, text_mask, labels)
        out_logit = llm_output.logits  # [B, L, C]

        if out_logit.isnan().any():
            logger.warning(
                f"NaN detected, ids: {ids}, text: {text}, visual_features:{visual_features.mean()}, out_logit:{out_logit.mean()}"
            )

        # out_loglogit = F.log_softmax(out_logit, dim=-1)
        #
        with torch.no_grad():
            # clip the last logits to calculate the accuracyk
            reduced_logits = out_logit[..., : self.llm_vocab_size]

        self.train_accu.update(
            rearrange(reduced_logits, "b l c -> (b l) c"),
            rearrange(labels, "b l -> (b l)"),
        )

        target_loss = llm_output.loss
        self.log("train_llm_generate_loss", target_loss, prog_bar=True)
        return target_loss

    def validation_step(self, batch, batch_idx):
        # forward visual features to avoid duplicated memory computation
        ids, video, video_length, text = self.dispatch_batch(batch, self.device)

        B = len(ids)

        outputs = self.generate(
            video=video,
            video_length=video_length,
            max_length=50,  # NOTE: max length in dev set is 50
        )

        for b in range(B):
            predicted = self.llm_tokenizer.decode(outputs[b], skip_special_tokens=True)
            self.bleu.update(
                [predicted],
                [[text[b]]],
            )

    def on_train_epoch_end(self):
        train_acc = self.train_accu.compute()
        self.log("train_llm_generate_accu", train_acc, prog_bar=True, sync_dist=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self):
        bleu = self.bleu.compute()
        self.log("val_llm_generate_bleu", bleu, prog_bar=True, sync_dist=True)
        self.bleu.reset()

    def train(self, is_train):
        super().train(is_train)
        self.shared_encoder.bert.embeddings.eval()
        self.visual_encoder.eval()

        self.visual_adapter.eval()
        # self.shared_encoder.eval()

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

    def load_from_bootstrap(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the model from a bootstrap state dict.
        """
        # NOTE: remove some of the keys connector
        for key in list(state_dict.keys()):
            if key.startswith("connector."):
                del state_dict[key]

        keys = self.load_state_dict(state_dict, strict=False)

        # NOTE: print information
        for key in keys.missing_keys:
            if not key.startswith("llm.") and not key.startswith("connector."):
                logger.warning(f"Missing key {key} in the state dict")
        for key in keys.unexpected_keys:
            logger.warning(f"Unexpected key {key} in the state dict")

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor = None,
        video_length: torch.Tensor = None,
        max_length: Optional[int] = 50,
    ):
        """
        @param video: (batch_size, seq_len, 3, h, w)
        """
        visual_embeddings, v_length = self.forward_visual(video, video_length)

        B = video.shape[0]

        visual_features = self._forward_q_former(visual_embeddings, v_length)

        prompt = self._prepare_llm_prompt(visual_features)

        _, PROMPT_LENGTH, _ = prompt.shape

        with torch.no_grad():
            # NOTE: max length in dev set is 50
            outputs = self.llm.generate(
                inputs_embeds=prompt,
                max_length=max_length + PROMPT_LENGTH,  # add the prompt length
                do_sample=False,
            )

        return outputs


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
