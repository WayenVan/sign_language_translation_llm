from transformers import MarianMTModel, MarianTokenizer

# 加载模型和 tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 输入文本
src_text = ["Hello, how are you?"]

# 分词并翻译
inputs = tokenizer(src_text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)

# 解码输出
tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(tgt_text)
