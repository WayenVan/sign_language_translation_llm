import torch
import sys
from transformers.cache_utils import DynamicCache

sys.path.append(".")
from modules.llamma_decoder.llamma_decoder import LlamaCrossDecoder

model = LlamaCrossDecoder(1024, 2048, 8, 8, 10000, 3).cuda()

input = torch.randint_like(torch.empty(2, 1), 0, 10000).cuda().long()
visual = torch.randn(2, 30, 1024).cuda()

cache = DynamicCache()

for i in range(8):
    output = model(input, visual, use_cache=True, past_key_values=cache)
    print(cache)

    input = output.last_hidden_state.argmax(dim=-1)
