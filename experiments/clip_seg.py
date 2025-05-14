from transformers import AutoTokenizer, AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image


tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").cuda()
processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")


image = Image.open("outputs/visualization_val/1.jpg").convert("RGB")

inputs = processor(
    text=["a photo of human moving his hands"],
    images=[image],
    return_tensors="pt",
    padding=True,
)

inputs = {k: v.cuda() for k, v in inputs.items()}

outputs = model(**inputs)
logits = outputs.logits
