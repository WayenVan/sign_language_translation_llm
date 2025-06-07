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
Your task is to comppress the following german text to a keywords list.
The rules are as follows:
1. keep original words and phrases, do not change the order of the words.
2. Retrain the original meaning of the text as much as possible.
3. The generated keywords list should be able to reconstruct the original text.
2. the formate should be like this: keywords1, keywords2, keywords3
3. do not include any other text, only the keywords list.
4. each keyword should only contain one word, do not include any special characters.
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
print(result)
outputs.write(result + "\n")
outputs.close()
