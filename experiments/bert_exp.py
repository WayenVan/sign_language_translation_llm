import sys
import os

sys.path.append(".")
from data.ph14t.ph14t_index import Ph14TIndex
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.models.bert import BertModel
from huggingface_hub import login

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

for name, module in model.named_modules():
    print(name)

for id in index.ids[:2]:
    data_info = index.get_data_info_by_id(id)
    translation = data_info["translation"]
    print("translation: " + translation)

    prompt = (
        "Translate the following german to english, responds with only the translation and nothng else. no explanation or punctuation outside the translation. \n\n"
        + translation
        + "\n"
        + "the translation is:\n"
    )
    output = tokenizer(translation)
    print("output: " + str(output))
