import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM
from huggingface_hub import login

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
login(token)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="cuda:0",
    torch_dtype=torch.float16,
).eval()

prompt = "hello words, this is a test. "
