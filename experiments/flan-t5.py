from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

for name, module in model.named_modules():
    print(name, module.__class__.__name__)

inputs = [
    "Reconstruct the following setences The cat sat on the mat.",
    "The dog chased the ball.",
    "The sun is shining brightly.",
]
labels = [
    "cat sat on the mat.",
    "dog chased the ball.",
    "sun is shining brightly.",
]


tokens = tokenizer.tokenize("The cat sat on the mat.", add_special_tokens=True)
inputs = tokenizer(
    inputs,
    text_target=labels,
    return_tensors="pt",
    padding="longest",
)
embeddings = model.get_input_embeddings()(inputs["input_ids"])  # [B, L, D]
out = model.encoder(
    inputs_embeds=embeddings,
)
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
