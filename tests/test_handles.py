import sys

sys.path.append(".")
from model.handles.vtm_handle import VTMHandle
from model.handles.vtg_handle import VTGHandle
import torch


def test_vtm_handle_mask():
    mask = VTMHandle.generate_padding_attention_mask(
        torch.tensor([5, 3]), torch.tensor([4, 2])
    )
    print(mask)
    print(mask.shape)  # Should be (2, 1, 1, 12)


def test_vtg_handle_mask():
    mask = VTGHandle.generate_padding_casual_attention_mask(
        torch.tensor([5, 3]), torch.tensor([4, 2])
    )
    print(mask)
    print(mask.shape)  # Should be (2,  1, 12)


if __name__ == "__main__":
    # test_handle()
    test_vtg_handle_mask()
