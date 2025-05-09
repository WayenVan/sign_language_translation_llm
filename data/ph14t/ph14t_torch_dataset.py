from torch.utils.data import Dataset
from .ph14t_index import Ph14TIndex
import cv2
import numpy
import os
import polars as pl


class Ph14KeywordDataset(Dataset):
    def __init__(
        self, data_root: str, keyword_dir: str, mode: str = "train", transforms=None
    ):
        self.ph14t_index = Ph14TIndex(data_root, mode)
        self.ids = self.ph14t_index.ids
        self.mode = mode
        self.transforms = transforms

        self.keyword_dir = keyword_dir
        self.keyword_file = os.path.join(
            self.keyword_dir, f"{self.mode}-extracted-keywords.csv"
        )
        self.keyword_data = pl.read_csv(self.keyword_file, separator="|")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        data_info = self.ph14t_index.get_data_info_by_id(id)

        video_frame_file_name = data_info["frame_files"]
        video_frame = []
        for frame_file in video_frame_file_name:
            image = cv2.imread(frame_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frame.append(image)

        keywords = self.keyword_data.filter(pl.col("id") == id).to_dicts()[0][
            "keywords"
        ]

        ret = dict(
            id=id,
            # NOTE: [time, height, width, channel], normalized to [0, 1]
            video=numpy.array(video_frame, dtype=numpy.float32) / 255.0,
            translation=data_info["translation"],
            keywords=keywords,
        )

        if self.transforms:
            ret = self.transforms(ret)
        return ret
