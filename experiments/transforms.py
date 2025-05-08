from kornia.augmentation import RandomResizedCrop
from torch.distributions import transforms
import torch
import einops


image = torch.randn(5, 18, 3, 224, 224)


class RandomCropVideo:
    """
    Random Crop Resize for video, not that the parameters used inside the video is the same  across the batch
    """

    def __init__(self, *args, **kwargs):
        self.crop = RandomResizedCrop(*args, **kwargs, same_on_batch=True)

    def __call__(self, data):
        video = data["video"]
        B, T, C, H, W = video.shape
        params = self.crop.generate_parameters([B, C, H, W])

        params["src"] = (
            params["src"].unsqueeze(1).expand(-1, T, -1, -1).contiguous().flatten(0, 1)
        )
        params["dst"] = (
            params["dst"].unsqueeze(1).expand(-1, T, -1, -1).contiguous().flatten(0, 1)
        )
        params["input_size"] = (
            params["input_size"]
            .unsqueeze(1)
            .expand(-1, T, -1)
            .contiguous()
            .flatten(0, 1)
        )
        params["output_size"] = (
            params["output_size"]
            .unsqueeze(1)
            .expand(-1, T, -1)
            .contiguous()
            .flatten(0, 1)
        )
        video = einops.rearrange(video, "b t c h w -> (b t) c h w")
        self.crop(video, params=params)
        video = einops.rearrange(video, "(b t) c h w -> b t c h w", b=B, t=T)
        return video


t = RandomCropVideo(size=(224, 224), scale=(0.5, 1.0), ratio=(0.75, 1.3333), p=1.0)

t({"video": image})
