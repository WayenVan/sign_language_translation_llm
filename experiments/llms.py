import polars as pl
from huggingface_hub import login
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
from transformers import AutoTokenizer


from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# 输入文本
#
model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda:1"
).eval()

processor: Gemma3Processor = AutoProcessor.from_pretrained(model_id)
# models.gemma3.Gemma3Processor
tokenizer = AutoTokenizer.from_pretrained(model_id)

kwds = open("outputs/outputs.txt", "r").readline()
text = [
    """
Reconstruct the  following german keywords to a sentence, \n
the rules are as follows: \n
1. The output should only contains the reconstructed sentence, do not include any other text. \n
2. Do not inlude any special characters, only include the words. \n
the keywrods are: \n
"""
    + kwds
]
inputs = tokenizer(
    text,
    return_tensors="pt",
)

embeds = model.get_input_embeddings()(inputs["input_ids"].to("cuda:1"))
outputs = model.generate(
    inputs_embeds=embeds,
    max_new_tokens=100,
    do_sample=False,
    # top_k=False,
)

# print(outputs.logits.shape)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
