"""
Generate augmented text for PHOENIX-2014-T dataset using DeepSeek
The estimated cost is abount 0.1 USD per 1000 samples.
"""

import polars as pl
import click
import os
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import logging


system_prompt = """
You are a professional german language specialist.
Your task is to performance a data augmentation task for the given text.
You should follow the instructions below:
1.Your should generate a new text that is similar to the given text, but with some variations.
2.Hoewever, you should strictly not change the meaning of the text.
3.Try to use the same words as much as possible but change structure of sentance.
4.Dot not add any markers or special characters, just the text. As it was in the original text.
5.Each sentence should genearte 5 variations, and each variation should be separated by a new line.
6.The output should only contain the generated text, without any additional text or markers.
7.Remove all punctuation marks, special characters from the text.
"""
api_key = os.getenv("DEEPSEEK_API_KEY")


def build_convertor():
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def convert(text):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": text},
            ],
            stream=False,
        )

        result = response.choices[0].message.content
        return result

    return convert


@click.command()
@click.option(
    "--ph14t_root",
    default="dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
)
@click.option(
    "--output_dir",
    default="outputs/ph14t_text_augmented/",
)
@click.option(
    "--num_workers",
    default=8,
)
@click.option(
    "--verify_mode",
    is_flag=True,
)
def main(ph14t_root, output_dir, num_workers, verify_mode):
    if verify_mode:
        verify_texts(output_dir, ph14t_root)
        return
    os.makedirs(output_dir, exist_ok=False)
    convertor = build_convertor()
    for mode in ["train", "dev", "test"]:
        with open(os.path.join(output_dir, "{}-augmented.csv".format(mode)), "w") as f:
            f.write("id|text\n")
            df = pl.read_csv(
                os.path.join(
                    ph14t_root,
                    "annotations/manual/PHOENIX-2014-T.{}.corpus.csv".format(mode),
                ),
                separator="|",
            )
            df = df.set_sorted("name")

            def process_single(id):
                translation = df.filter(df["name"] == id)["translation"].item()
                respones = convertor(translation)
                auged_text = respones.split("\n")

                ret = []
                for t in auged_text:
                    cleaned = re.sub(r"[^\w\s]", "", t)  # 去除标点符号等特殊符号
                    cleaned = re.sub(
                        r"\s+", " ", cleaned
                    )  # 将多个空白（空格、\n、\t 等）替换为一个空格
                    cleaned = cleaned.strip()

                    if len(cleaned) < 5:
                        continue

                    ret.append(id + "|" + cleaned)
                return "\n".join(ret)

            futures = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for id in df["name"]:
                    future = executor.submit(process_single, id)
                    futures.append(future)

                i = 0
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing"
                ):
                    try:
                        result = future.result()
                        print(result)
                        f.write(result + "\n")
                        if i % 50 == 0:
                            f.flush()
                        i += 1

                    except Exception as e:
                        print("Error:", e)


def verify_texts(output_dir, ph14t_root):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(output_dir + "/verify_log.log", mode="w"))
    logger.addHandler(logging.StreamHandler())

    for mode in ["train", "dev", "test"]:
        df_extended = pl.read_csv(
            os.path.join(output_dir, "{}-augmented.csv".format(mode)),
            separator="|",
        )
        df = pl.read_csv(
            os.path.join(
                ph14t_root,
                "annotations/manual/PHOENIX-2014-T.{}.corpus.csv".format(mode),
            ),
            separator="|",
        )
        df = df.set_sorted("name")

        for id in df["name"].unique():
            texts = df_extended.filter(df_extended["id"] == id)["text"].to_list()
            if len(texts) == 0:
                logger.error(f"ID {id} has no variations.")
                continue
            if len(texts) < 5:
                logger.warning(f"ID {id} has less than 5 variations: {len(texts)}")
            elif len(texts) > 5:
                logger.warning(f"ID {id} has more than 5 variations: {len(texts)}")


if __name__ == "__main__":
    main()
