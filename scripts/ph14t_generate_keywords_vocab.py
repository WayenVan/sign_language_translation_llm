import numpy as np
import polars as pl
from transformers import AutoTokenizer
import click
import os
from huggingface_hub import login

hg_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
login(hg_token)
# model_id = "google/gemma-3-4b-it"
model_id = "google/gemma-3-12b-it"


@click.command()
@click.option(
    "--keyword_dir",
    default="outputs/keywords",
)
@click.option(
    "--output_dir",
    default="outputs",
)
def main(keyword_dir, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = set()
    for mode in ["train", "dev", "test"]:
        file_name = f"{mode}-extracted-keywords.csv"
        df = pl.read_csv(
            os.path.join(keyword_dir, file_name),
            separator="|",
        )
        for keywords in df["keywords"]:
            if len(keywords) > 0:
                # keywords = keywords.split(" ")
                tokens = tokenizer.tokenize(keywords)
                vocab.update(tokens)

    vocab = sorted(list(vocab))
    with open(
        os.path.join(output_dir, "keywords_vocab.txt"), "w", encoding="utf-8"
    ) as f:
        for word in vocab:
            f.write(word + "\n")


if __name__ == "__main__":
    main()
