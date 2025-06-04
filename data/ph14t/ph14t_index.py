import polars as pl
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from diskcache import Cache
import logging

logger = logging.getLogger(__name__)


class Ph14TIndex:
    """
    Index for the PHOENIX-2014T dataset.
    using polars for efficient data handling and parallel processing.
    generates a frame file table by multithreading to speed up the process.
    """

    cache = Cache(".cache/ph14t_index")

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

        self.frame_file_table = self._load_or_generate_frame_file_table(
            self.feature_root
        )

    def _load_or_generate_frame_file_table(self, feature_root: str):
        abs_feature_root = os.path.abspath(feature_root)
        cache_key = f"frame_file_table_{abs_feature_root}_{self.mode}"

        if cache_key in self.cache:
            logger.info("Uinsg cached frame file table.")
            return self.cache[cache_key]
        else:
            logger.info("No cached frame file table found, generating new one.")
            frame_file_table = self._generate_frame_file_table()
            self.cache[cache_key] = frame_file_table
            return frame_file_table

    def _generate_frame_file_table(self):
        # Process files in parallel with dynamic chunking
        workers = min(32, (os.cpu_count() or 1) * 4)
        chunk_size = max(1, len(self.ids) // (workers * 4))  # Dynamic chunking

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Process batches of IDs in parallel
            futures = []
            for i in range(0, len(self.ids), chunk_size):
                batch = self.ids[i : i + chunk_size]
                futures.append(executor.submit(self._process_id_batch, batch))

            # Collect results with progress bar
            all_data = []
            for future in tqdm(futures, desc="Processing frames"):
                all_data.extend(future.result())

        # Single DataFrame creation
        return pl.DataFrame(
            all_data,
            schema={
                "id": pl.Utf8,
                "frame_file": pl.Utf8,
                "frame_index": pl.Int64,
            },
        )

    def _process_id_batch(self, id_batch):
        batch_data = []
        for id in id_batch:
            for frame_file in self._get_video_frame_file_name(id):
                batch_data.append(
                    {
                        "id": id,
                        "frame_file": frame_file,
                        "frame_index": int(frame_file[-8:-4]),
                    }
                )
        return batch_data

    def _get_video_frame_file_name(self, id: str):
        dir_path = os.path.join(self.feature_root, id)
        with os.scandir(dir_path) as it:
            return [
                os.path.join(dir_path, entry.name)
                for entry in it
                if entry.name.endswith(".png") and entry.is_file()
            ]

    def get_data_info_by_id(self, id: str):
        if id not in self.ids:
            raise ValueError(f"ID {id} not found in the dataset.")
        data_info = self.raw_annotation.filter(pl.col("name") == id).to_dicts()[0]

        selected = self.frame_file_table.filter(pl.col("id") == id)
        framefiles = selected.sort("frame_index")["frame_file"].to_list()
        if not framefiles:
            raise ValueError(f"No frame files found for ID {id}.")
        data_info["frame_files"] = framefiles

        return data_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_root = "dataset/PHOENIX-2014-T-release-v3/"
    ph14t_index = Ph14TIndex(data_root, "train")
    print(ph14t_index.frame_file_table)

    for ph14t_id in ph14t_index.ids[:10]:
        data_info = ph14t_index.get_data_info_by_id(ph14t_id)
        print(data_info)
