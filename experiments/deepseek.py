# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import os
import polars as pl

from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention


outputs = open("outputs/outputs.txt", "w")

api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

df = pl.read_csv(
    "dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv",
    separator="|",
)

system_prompt = """
You are a professional german language specialist.
Your task is to performance a data augmentation task for the given text.
Your should generate a new text that is similar to the given text, but with some variations.
Hoewever, you should strictly not change the meaning of the text.
Try to use the same words as much as possible but change structure of sentance.
Dot not add any markers or special characters, just the text. As it was in the original text.
Each sentence should genearte 5 variations, and each variation should be separated by a new line.
The output should only contain the generated text, without any additional text or markers.
Remove all punctuation marks, special characters from the text.

"""

text = df[3, -1]
print(text)
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

results = result.split("\n")
outputs.close()

for r in results:
    print(r)
