import polars as pl
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mistral3 import Mistral3ForConditionalGeneration
import os
import torch


model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
huggingface_hub_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
login(huggingface_hub_token)

from huggingface_hub import HfFileSystem

#
model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
).eval()

embed_layer = model.get_input_embeddings().cuda()
del model
print(embed_layer)
