from lightning import LightningDataModule
import torch
from torch.nn import functional as F
from data.ph14t.ph14t_torch_dataset import Ph14TDataset
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np

from typing import List


class Ph14TDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,  # NOTE: any keyword arguments for DataLoader
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.batch_size_val_test = cfg.data.batch_size_val_test

        self.transforms_train = instantiate(getattr(cfg.data.transforms, "train", None))
        self.transforms_val = instantiate(getattr(cfg.data.transforms, "val", None))
        self.transforms_test = instantiate(
            getattr(cfg.data.transforms, "test", self.transforms_val),
        )
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Set up the dataset for training, validation, and testing
        if stage == "fit" or stage is None:
            self.train_dataset = Ph14TDataset(
                data_root=self.cfg.data.data_root,
                mode="train",
                transforms=self.transforms_train,
            )
            self.val_dataset = Ph14TDataset(
                data_root=self.cfg.data.data_root,
                mode="dev",
                transforms=self.transforms_val,
            )
        if stage == "test" or stage is None:
            self.test_dataset = Ph14TDataset(
                data_root=self.cfg.data.data_root,
                mode="test",
                transforms=self.transforms_test,
            )

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            **self.kwargs,
        )

    def val_dataloader(self):
        # Return the validation dataloader
        kwargs = self.kwargs.copy()
        kwargs["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val_test,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def test_dataloader(self):
        # Return the test dataloader
        kwargs = self.kwargs.copy()
        kwargs["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_val_test,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch):
        videos = [
            torch.from_numpy(item["video"])
            if isinstance(item["video"], np.ndarray)
            else item["video"]
            for item in batch
        ]
        v_length = [video.shape[0] for video in videos]
        max_v_length = max(v_length)
        padded_videos = [
            F.pad(video, (0, 0, 0, 0, 0, 0, 0, max_v_length - video.shape[0]))
            for video in videos
        ]
        padded_videos = torch.stack(padded_videos)

        ret = dict(video=padded_videos)

        # handle other keys in the batch
        for key in batch[0].keys():
            if key == "video":
                continue
            if isinstance(batch[0][key], torch.Tensor):
                ret[key] = torch.stack([item[key] for item in batch])
            elif (
                isinstance(batch[0][key], List)
                or isinstance(batch[0][key], str)
                or isinstance(batch[0][key], tuple)
            ):
                ret[key] = [item[key] for item in batch]

        return ret
