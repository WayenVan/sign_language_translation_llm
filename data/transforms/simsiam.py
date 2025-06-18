import torch
import os

import albumentations as A
import numpy as np


def to_tensor(video):
    video = torch.tensor(video, dtype=torch.float32)
    video = video.permute(
        0, 3, 1, 2
    )  # [time, height, width, channel] -> [time, channel, height, width]
    video = video.contiguous()
    return video


def sample_continuous_clip(video, n_frames):
    """
    Sample a continuous clip of N frames from a video.

    Args:
        video: numpy array with shape (time, height, width, channel)
        n_frames: number of frames to sample

    Returns:
        A clip of shape (n_frames, height, width, channel)
    """
    t = video.shape[0]

    if t >= n_frames:
        start = np.random.randint(0, t - n_frames + 1)
        clip = video[start : start + n_frames]
    else:
        # Pad with last frame if video is shorter than n_frames
        last_frame = video[-1:]
        padding = np.tile(last_frame, (n_frames - t, 1, 1, 1))
        clip = np.concatenate([video, padding], axis=0)

    return clip


class SimSiamTransformForTrain:
    def __init__(
        self,
        sample_n_frames=32,
        crop_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=1.0,
        flip_p=0.5,
    ):
        self.transform = A.Compose(
            [
                A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value),
            ]
        )
        self.sample_n_frames = sample_n_frames

    def __call__(self, data):
        video = data["video"]
        video = sample_continuous_clip(video, self.sample_n_frames)
        data["video"] = video
        video_aug1 = self.transform(images=video)["images"]
        video_aug2 = self.transform(images=video)["images"]

        video_aug1 = to_tensor(video_aug1)
        video_aug2 = to_tensor(video_aug2)

        data["video_aug1"] = video_aug1
        data["video_aug2"] = video_aug2
        return data


class SimSiamTransformForEval:
    def __init__(
        self,
        sample_n_frames=32,
        crop_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=1.0,
    ):
        self.transform = A.Compose(
            [
                A.CenterCrop(height=crop_size[0], width=crop_size[1], p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value),
            ]
        )
        self.sample_n_frames = sample_n_frames

    def __call__(self, data):
        video = data["video"]
        video = sample_continuous_clip(video, self.sample_n_frames)
        data["video"] = video
        video_aug = self.transform(images=video)["images"]
        video_aug = to_tensor(video_aug)
        data["video_aug1"] = video_aug
        data["video_aug2"] = video_aug
        return data
