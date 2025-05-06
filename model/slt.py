from huggingface_hub.inference._generated.types import video_classification
from numpy.polynomial.polyutils import as_series
import torch
from torch import nn
from lightning import LightningDataModule, LightningModule
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from omegaconf import DictConfig
from transformers.models.mistral3 import Mistral3ForConditionalGeneration
from hydra.utils import instantiate
from torchmetrics import Accuracy


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

        self.visual_backbone = None
        self.encoder = None
        self.decoder = None

        self.loss = None

        self.train_accu = Accuracy(task="multiclass", num_classes=len(vocab))

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

        if self.debug:
            print(f"tokens: {sentence}")
        return ids

    def forward(
        self,
        x,
        video_length=None,
    ):
        """
        Forward pass through the model.
        """

        visual_features, v_length = self.video_backbone(x, video_length)
        encoded_features = self.encoder(visual_features, v_length)
        decoded_features = self.decoder.generate(visual_features, v_length)
        return decoded_features

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
        keywords_ids_out, _, _ = self._pad_tokens_for_decoder(keywords_ids_out)

        keywords_llm_in = self.llm_tokenizer(
            keywords, return_tensors="pt", padding=True
        )

        # assert the valid range of tokens between llms and my decoder should be the same
        for i in range(len(keywords_ids_in)):
            assert mask[i].cpu().tolist() == keywords_llm_in[i].attention_mask

        return (
            keywords_ids_in,
            keywords_llm_in,
            keywords_ids_out,
            mask,
            keywords_lengths,
        )

    @torch.no_grad()
    def prepare_for_token_level_accuracy(
        self, logits, keywords_ids_out, padding_mask, keywords_lengths
    ):
        """
        Prepare the logits and target IDs for token-level accuracy calculation.
        """
        # Flatten the logits and target IDs
        logits = logits.flatten(0, 1)
        target_ids = keywords_ids_out.flatten()

        # Get the available logits and target IDs based on the padding mask
        available_logits = logits[padding_mask.flatten() == 1, :]
        available_target_ids = target_ids[padding_mask.flatten() == 1]

        return available_logits, available_target_ids

    def training_step(self, batch, batch_idx):
        names = batch["names"]
        keywords = batch["keywords"]
        video = batch["video"]
        video_length = batch["video_length"]

        (
            keywords_ids_in,
            keywords_llm_in,
            keywords_ids_out,
            padding_mask,
            keywords_lengths,
        ) = self.preprocess_train_keywords(keywords)

        visaul_outputs = self.visual_backbone(video, video_length)
        encoder_outputs = self.encoder(visaul_outputs)
        decoder_outputs = self.decoder(
            encoder_outputs,
            decoder_input_ids=keywords_ids_in,
            padding_mask=padding_mask,
            keywords_lengths=keywords_lengths,
        )
        loss_outputs = self.loss(
            decoder_outputs,
            keywords_ids_out,
            keywords_lengths=keywords_lengths,
            padding_mask=padding_mask,
        )

        # Calculate the logits and target IDs for token-level accuracy
        avialiable_predicted_ids, avialiable_target_ids = (
            self.prepare_for_token_level_accuracy(
                decoder_outputs.logits,
                keywords_ids_out,
                padding_mask,
                keywords_lengths,
            )
        )
        self.train_accu.update(avialiable_predicted_ids, avialiable_target_ids)

        # Log the loss
        for loss_name in loss_outputs.as_dict():
            self.log(loss_name, loss_outputs[loss_name], prog_bar=True)

        return loss_outputs.loss

    def on_train_epoch_end(self, outputs):
        """
        Called at the end of each training epoch.
        """
        # Calculate and log the accuracy
        train_acc = self.train_accu.compute()
        self.log("train_token_level_accu", train_acc, prog_bar=True)
        self.train_accu.reset()

    def validation_step(self, batch, batch_idx):
        pass

    def train(self, is_train):
        """
        Override the train method to set the model to training mode.
        """
        super().train(is_train)
        self.llm_embedding_layer.eval()  # always eval for llm embedding layer


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
