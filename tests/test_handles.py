import sys

sys.path.append(".")
from model.handles.mlm_handle import MLMHandle
import torch


def test_handle():
    mask = MLMHandle.generate_padding_attention_mask(
        torch.tensor([5, 3]), torch.tensor([4, 2])
    )
    print(mask)
    print(mask.shape)  # Should be (2, 1, 1, 12)


if __name__ == "__main__":
    test_handle()
