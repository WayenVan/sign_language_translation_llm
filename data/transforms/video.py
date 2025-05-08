import torch
from kornia.augmentation import RandomCrop3D, Resize, Normalize, CenterCrop3D
from torchvision import transforms


class RandomCrop3DVideo:
    def __init__(self, *args, **kwargs):
        self.crop = RandomCrop3D(*args, **kwargs)

    def __call__(self, data):
        video = data["video"]
        video = self.crop(video)
        data["video"] = video
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
