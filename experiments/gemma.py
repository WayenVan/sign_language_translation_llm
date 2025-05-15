import sys
import os

sys.path.append(".")
from data.ph14t.ph14t_index import Ph14TIndex
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM
from huggingface_hub import login

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
login(token)


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

    prompt = (
        "Translate the following german to english, responds with only the translation and nothng else. no explanation or punctuation outside the translation. \n\n"
        + translation
        + "\n"
        + "the translation is:\n"
    )
    output = tokenizer.tokenize(translation)
    idx = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    embeddings = model.get_input_embeddings()
    ebd = embeddings(idx["input_ids"])

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=ebd,
            max_length=256,
            do_sample=False,
            # top_k=50,
            # top_p=0.95,
            # temperature=0.7,
        )

    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("output: " + output)
