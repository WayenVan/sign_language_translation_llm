import sys

sys.path.append(".")
from model.handles.vtm_handle import VTMHandle
from model.handles.vtg_handle import VTGHandle
from model.handles.vtc_handle import VTCHandle
import torch


def test_vtm_handle_mask():
    mask = VTMHandle.generate_padding_attention_mask(
        5, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    )
    print(mask)
    print(mask.shape)  # Should be (2, 1, 1, 12)


def test_vtg_handle_mask():
    mask = VTGHandle.generate_padding_casual_attention_mask(
        5, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    )
    print(mask)
    print(mask.shape)  # Should be (2,  1, 12)


def test_vtc_handle_mask():
    mask = VTCHandle.generate_padding_casual_attention_mask(
        5, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    )
    print(mask)
    print(mask.shape)  # Should be (2,  1, 12)


if __name__ == "__main__":
    # test_handle()
    # test_vtg_handle_mask()
    # test_vtm_handle_mask()
    test_vtc_handle_mask()
