import polars as pl
from huggingface_hub import login
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
from transformers import AutoTokenizer


df = pl.read_csv(
    "dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv",
    separator="|",
)

model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# open print the tokenized result
# 输入文本
text = df[3, -1]
print(text)

# Tokenize 文本（返回 input_ids 和 attention_mask）
inputs = tokenizer(text, return_tensors="pt")

# 获取 token IDs
token_ids = inputs["input_ids"][0]  # 取第一个（唯一）序列
print(token_ids)

# 转换为 tokens（即 label）
tokens = tokenizer.convert_ids_to_tokens(token_ids)

# 打印 tokens
print(tokens)
"""
result: ['[CLS]', 'hello', ',', 'how', 'are', 'you', '?', '[SEP]']
"""
