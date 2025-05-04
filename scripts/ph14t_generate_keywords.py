import polars as pl
import click
import os
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


system_prompt = """
You are a professional german language specialist.
Your task is to comppress the following german text to a keywords list.
The rules are as follows:
1. keep original words and phrases, do not change the order of the words.
2. Retrain the original meaning of the text as much as possible.
3. The generated keywords list should be able to reconstruct the original text.
2. Every keyword should speperated by a space, the formate should be like this: keywords1 keywords2 keywords3
3. do not include any other text, only the keywords list.
4. each keyword should only contain one word, do not include any special characters.
"""
api_key = os.getenv("DEEPSEEK_API_KEY")


def build_keywords_convertor():
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
    default="outputs/keywords/",
)
@click.option(
    "--num_workers",
    default=8,
)
def main(ph14t_root, output_dir, num_workers):
    convertor = build_keywords_convertor()
    for mode in ["train", "dev", "test"]:
        with open(
            os.path.join(output_dir, "{}-extracted-keywords.csv".format(mode)), "w"
        ) as f:
            f.write("id|keywords|translation\n")
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
                keywrds = convertor(translation)
                return id + "|" + keywrds + "|" + translation

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


if __name__ == "__main__":
    main()
