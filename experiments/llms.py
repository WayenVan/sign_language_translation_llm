import polars as pl
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mistral3 import Mistral3ForConditionalGeneration
import os
import torch
import numpy as np

print(torch.cuda.device_count())


huggingface_hub_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
login(huggingface_hub_token)

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# 输入文本
#
model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
).eval()

# models.gemma3.Gemma3Processor
tokenizer = AutoTokenizer.from_pretrained(model_id)

for i in range(100):
    df = pl.read_csv(
        "outputs/keywords/test-extracted-keywords.csv",
        separator="|",
    )
    kwds = df[i, -2]
    translate = df[i, -1]
    print("keywords: " + kwds)
    print("translation: " + translate)
    print("----------------------------------------------")

    text = [
        """
    Reconstruct the  following german keywords to a sentence, \n
    the rules are as follows: \n
    1. each keyword is separated by a space \n
    2. the sentence should be a grammatically correct sentence \n
    3. the sentence should be a meaningful sentence \n
    4. the sentence should be a complete sentence \n
    5. you can add some words to the sentence, but you should not remove any words \n
    6. all characters should be in lowercase \n
    7. only one sentence should be generated \n
    8. The output should only contains the reconstructed sentence, do not include any other text. \n
    9. The output should be in one line, do not include any line breaks \n

    For example: \n
    Input: \n
    im süden zeigt sonne \n
    Output: \n
    im süden zeigt sich aber auch die sonne \n

    the keywrods are: \n
    """
        + kwds
        + "\n"
        + """
    The output sentence is: \n
    """
    ]
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    with torch.inference_mode():
        embeds = model.get_input_embeddings()(inputs["input_ids"].cuda())
        outputs = model.generate(
            inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"].cuda(),
            max_new_tokens=100,
            do_sample=False,
            # top_k=False,
        )

    # print(outputs.logits.shape)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)
    print("-------------------------------------------------")
