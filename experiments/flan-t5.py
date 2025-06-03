from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

inputs = tokenizer(
    "translate  to german: today's weather seems not good",
    return_tensors="pt",
)
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
