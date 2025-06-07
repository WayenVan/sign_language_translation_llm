import sys
import torch


sys.path.append(".")

from modules.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from transformers import FSMTTokenizer


mname = "WayenVan/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = [  #
    "The world of machine learning is fascinating, isn't it?",
    "The world of machine learning is fascinating, isn't it?",
]

eos_token_id = tokenizer.encoder["</s>"]
bos_token_id = tokenizer.encoder["</s>"]


text_ids = [tokenizer.encode(text, add_special_tokens=False) for text in input]
input_ids = [[bos_token_id] + text_id for text_id in text_ids]
label_ids = [text_id + [eos_token_id] for text_id in text_ids]

input_ids = tokenizer.pad(
    {"input_ids": input_ids},
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)
label_ids = tokenizer.pad(
    {"input_ids": label_ids},
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)

eos_token_id = tokenizer.encode("", return_tensors="pt")
input_ids = tokenizer.encode(input, return_tensors="pt", add_special_tokens=False)
encoder_embeddings = model.get_input_embeddings()(
    torch.cat([input_ids, eos_token_id], dim=-1)
)

outputs = model.generate(inputs_embeds=encoder_embeddings)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)  # Maschinelles Lernen ist großartig, oder?
