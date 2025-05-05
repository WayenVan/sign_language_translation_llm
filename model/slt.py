from huggingface_hub.inference._generated.types import video_classification
import torch
from torch import device, nn
from lightning import LightningDataModule, LightningModule
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from omegaconf import DictConfig
from transformers.models.mistral3 import Mistral3ForConditionalGeneration
from hydra.utils import instantiate

print(torch.cuda.is_bf16_supported())


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
        self.padding_idx = self.reverse_vocab[self.llm_token.pad_token]

        self.visual_backbone = None
        self.encoder = None
        self.decoder = None

    def _create_tokenizer(self):
        self.llm_token = AutoTokenizer.from_pretrained(self.llm_id)

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

    def sentence_to_ids(self, sentence, add_bos=False, add_eos=False):
        """
        Convert a sentence to a list of token IDs.
        """
        tokens = self.llm_token.tokenize(sentence)
        if add_bos:
            tokens = [self.llm_token.bos_token] + tokens

        if add_eos:
            tokens = tokens + [self.llm_token.eos_token]

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

        keywords_ids_in, attention_mask, keywords_lengths = (
            self._pad_tokens_for_decoder(keywords_ids_in)
        )
        keywords_ids_out, _, _ = self._pad_tokens_for_decoder(keywords_ids_out)

        keywords_ids_in_llm = self.llm_token(
            keywords, return_tensors="pt", padding=True
        )

        return (
            keywords_ids_in,
            keywords_ids_in_llm,
            keywords_ids_out,
            attention_mask,
            keywords_lengths,
        )

    def training_step(self, batch, batch_idx):
        names = batch["names"]
        keywords = batch["keywords"]
        video = batch["video"]
        video_length = batch["video_length"]
        keywords_ids_in, keywords_ids_out, attention_mask, keywords_lengths = (
            self.preprocess_train_keywords(keywords)
        )
        return None

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"])
        val_loss = outputs.loss
        self.log("val_loss", val_loss)

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
        keyword_ids_in_llm,
        keyword_ids_out,
        attention_mask,
        keywords_lengths,
    ) = model.preprocess_train_keywords(idx)
    print("keyword_ids_in", keyword_ids_in)
