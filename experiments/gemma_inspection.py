import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM
from huggingface_hub import login

# token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
# login(token)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="auto",
    torch_dtype=torch.float16,
).eval()

prompt = "translate the following sentence to german, only produce the translated sentence: 'The world of machine learning is fascinating, isn't it?'"


inputs = tokenizer(prompt, return_tensors="pt")


output = model.generate(
    inputs["input_ids"].to("cuda:0"),
    attention_mask=inputs["attention_mask"].to("cuda:0"),
    max_new_tokens=50,
)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("output: " + output_text)
