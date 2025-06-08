import sys
import os

sys.path.append(".")
from data.ph14t.ph14t_index import Ph14TIndex
import torch

# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from huggingface_hub import login

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig


# token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
# login(token)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

index = Ph14TIndex("dataset/PHOENIX-2014-T-release-v3")

model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="cuda:0",
    torch_dtype=torch.float16,
).eval()


for id in index.ids[:2]:
    data_info = index.get_data_info_by_id(id)
    translation = data_info["translation"]
    print("translation: " + translation)

    prompt = "hello words, this is a test. "
    output = tokenizer(translation)
    idx = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    embeddings = model.get_input_embeddings()
    ebd = embeddings(idx["input_ids"])
    attnion_mask = idx["attention_mask"]

    with torch.no_grad():
        output = model(
            inputs_embeds=ebd,
        )

    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("output: " + output)
