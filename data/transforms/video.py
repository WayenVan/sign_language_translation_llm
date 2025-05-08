import torch
from kornia.augmentation import (
    Resize,
    Normalize,
    CenterCrop,
    RandomResizedCrop,
)


class RandomResizedCropVideo:
    """
    Random Crop Resize for video, not that the parameters used inside the video is the same  across the batch
    """

    def __init__(self, size, scale, ratio, p=1.0):
        self.crop = RandomResizedCrop(
            tuple(size), scale=tuple(scale), ratio=tuple(ratio), p=p, same_on_batch=True
        )

    def __call__(self, data):
        video = data["video"]
        data["video"] = self.crop(video)
        return data


class CenterCropVideo:
    """
    Center Crop Resize for video, not that the parameters used inside the video is the same  across the Temporal dimension
    """

    def __init__(self, size, p=1.0):
        self.crop = CenterCrop(tuple(size), p=p, same_on_batch=True)

    def __call__(self, data):
        video = data["video"]
        data["video"] = self.crop(video)
        return data


class ResizeVideo:
    def __init__(self, *args, **kwargs):
        self.resize = Resize(*args, **kwargs)

    def __call__(self, data):
        video = data["video"]
        video = self.resize(video)
        data["video"] = video
        return data


class NormalizeVideo:
    def __init__(self, *args, **kwargs):
        self.normalize = Normalize(*args, **kwargs)

    def __call__(self, data):
        video = data["video"]
        video = self.normalize(video)
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
