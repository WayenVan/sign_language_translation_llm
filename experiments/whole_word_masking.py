import sys
import os

sys.path.append(".")
from data.ph14t.ph14t_index import Ph14TIndex
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.models.bert import BertModel
from huggingface_hub import login
from transformers import DataCollatorForWholeWordMask

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
login(token)


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-europeana-cased")

index = Ph14TIndex("dataset/PHOENIX-2014-T-release-v3")

model = BertModel.from_pretrained(
    "dbmdz/bert-base-german-europeana-cased",
    device_map="cuda:0",
    torch_dtype=torch.float16,
).eval()

config = AutoConfig.from_pretrained(
    "dbmdz/bert-base-german-europeana-cased",
)

collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.9,
)

sentences = ["Playing football is fun", "Transformers are powerful models"]

# 第一步：tokenize（批量处理）
batch_encoding = [tokenizer(sentence) for sentence in sentences]
batch_encoding = collator(batch_encoding)
print(batch_encoding)
