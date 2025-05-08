import polars as pl
import numpy as np
import os
import re
import glob


class Ph14TIndex:
    def __init__(self, data_root: str, mode: str = "train"):
        self.data_root = data_root
        self.mode = mode

        self.feature_root = os.path.join(
            data_root, "PHOENIX-2014-T/features/fullFrame-210x260px", self.mode
        )

        self.raw_annotation = pl.read_csv(
            os.path.join(
                self.data_root,
                f"PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{self.mode}.corpus.csv",
            ),
            separator="|",
        )
        self.ids = self.raw_annotation["name"].to_list()

    def get_data_info_by_id(self, id: str):
        if id not in self.ids:
            raise ValueError(f"ID {id} not found in the dataset.")
        data_info = self.raw_annotation.filter(pl.col("name") == id).to_dicts()[0]

        glob_p = os.path.join(
            self.feature_root,
            f"{id}/*.png",
        )
        video_frame_file_name = glob.glob(glob_p)
        video_frame_file_name = sorted(
            video_frame_file_name, key=lambda x: int(x[-8:-4])
        )

        data_info["frame_files"] = video_frame_file_name
        return data_info


if __name__ == "__main__":
    data_root = "/root/projects/slt_set_llms/dataset/PHOENIX-2014-T-release-v3"
    ph14t_index = Ph14TIndex(data_root, "train")

    for ph14t_id in ph14t_index.ids[:10]:
        data_info = ph14t_index.get_data_info_by_id(ph14t_id)
        print(data_info)
