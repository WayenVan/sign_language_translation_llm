import torch
import sys
from transformers.cache_utils import DynamicCache

sys.path.append(".")
from modules.llamma_decoder.llamma_decoder import LlamaCrossDecoder


def test_decoder():
    model = LlamaCrossDecoder(1024, 2048, 8, 8, 10000, 3, 2, 1).cuda()

    input = torch.randint_like(torch.empty(2, 1), 0, 10000).cuda().long()
    visual = torch.randn(2, 30, 1024).cuda()

    cache = DynamicCache()

    for i in range(8):
        output = model(input, visual, use_cache=True, past_key_values=cache)
        print(cache)

        input = output.last_hidden_state.argmax(dim=-1)


def test_decoder_generate():
    from einops import repeat

    model = LlamaCrossDecoder(1024, 4096, 2048, 8, 8, 10000, 3, 2, 1).cuda()

    visual = torch.randn(30, 1024).cuda()
    visual = repeat(visual, "n d -> b n d", b=3)

    # the output should be the same
    output = model.generate(visual, max_length=10)
    print(output)


def test_decoder_generate_with_mask():
    from einops import repeat

    model = LlamaCrossDecoder(1024, 4096, 2048, 8, 8, 10000, 3, 2, 1).cuda()

    visual = torch.randn(10, 1024).cuda()
    visual = repeat(visual, "n d -> b n d", b=4)
    visual_padding_mask = (
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        .bool()
        .cuda()
    )

    # the output should be the same
    output = model.generate(
        visual, visual_padding_mask=visual_padding_mask, max_length=10
    )
    print(output)


if __name__ == "__main__":
    test_decoder_generate_with_mask()
