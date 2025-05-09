import torch
from kornia.augmentation import (
    Resize,
    Normalize,
    CenterCrop,
    RandomResizedCrop,
)
from abc import ABC, abstractmethod
import os

import albumentations as A


class RandomCropVideo:
    """
    Random Crop Resize for video, not that the parameters used inside the video is the same  across the batch
    """

    def __init__(self, *args, **kwargs):
        self.crop = A.RandomCrop(*args, **kwargs)

    def __call__(self, data):
        video = data["video"]
        data["video"] = self.crop(images=video)["images"]
        return data


class CenterCropVideo:
    """
    Center Crop Resize for video, not that the parameters used inside the video is the same  across the Temporal dimension
    """

    def __init__(self, *args, **kwargs):
        self.crop = A.CenterCrop(*args, **kwargs)

    def __call__(self, data):
        video = data["video"]
        data["video"] = self.crop(images=video)["images"]
        return data


class ResizeVideo:
    def __init__(self, *args, **kwargs):
        self.resize = A.Resize(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, data):
        video = data["video"]
        video = self.resize(images=video)["images"]
        data["video"] = video
        return data


class NormalizeVideo:
    def __init__(self, *args, **kwargs):
        self.normalize = A.Normalize(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, data):
        video = data["video"]
        video = self.normalize(images=video)["images"]
        data["video"] = video
        return data


class ToTensorVideo:
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        video = data["video"]
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(
            0, 3, 1, 2
        )  # [time, height, width, channel] -> [time, channel, height, width]
        video = video.contiguous()
        data["video"] = video
        return data


class ToGpuVideo:
    # WARN: still buggy, will cause the video data become None, unexpected behaviour
    def __call__(self, data):
        local_rank = os.getenv("LOCAL_RANK", None)

        if local_rank is not None:
            data["video"] = data["video"].to(f"cuda:{local_rank}")
            assert data["video"] is not None
            # print(data["video"].shape)
        else:
            raise RuntimeError(
                "LOCAL_RANK is not set. Please set it to the local rank of the process."
            )
