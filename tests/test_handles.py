import sys

sys.path.append(".")
from model.handles.mlm_handle import MLMHandle
from model.handles.itc_handle import ITCHandle
import torch


def test_mlm_handle_mask():
    mask = MLMHandle.generate_padding_attention_mask(
        torch.tensor([5, 3]), torch.tensor([4, 2])
    )
    print(mask)
    print(mask.shape)  # Should be (2, 1, 1, 12)


def test_itc_handle_mask():
    mask = ITCHandle.generate_padding_casual_attention_mask(
        torch.tensor([5, 3]), torch.tensor([4, 2])
    )
    print(mask)
    print(mask.shape)  # Should be (2, 1, 1, 12)


if __name__ == "__main__":
    # test_handle()
    test_itc_handle_mask()
