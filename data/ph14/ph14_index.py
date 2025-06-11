import polars as pl
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from diskcache import Cache
import logging

logger = logging.getLogger(__name__)


class Ph14Index:
    """
    Index for the PHOENIX-2014T dataset.
    using polars for efficient data handling and parallel processing.
    generates a frame file table by multithreading to speed up the process.
    """

    cache = Cache(".cache/ph14_index")

    def __init__(self, data_root: str, mode: str = "train", force_reload: bool = False):
        self.data_root = data_root
        self.mode = mode
        self.force_reload = force_reload

        self.feature_root = os.path.join(
            data_root,
            "phoenix-2014-multisigner/features/fullFrame-210x260px",
            self.mode,
        )
        self.frame_anno_table = self._load_frame_anno_table()

        self.raw_annotation = pl.read_csv(
            os.path.join(
                self.data_root,
                f"phoenix-2014-multisigner/annotations/manual/{self.mode}.corpus.csv",
            ),
            separator="|",
        )
        self.ids = self.raw_annotation["id"].to_list()

        self.frame_file_table = self._load_or_generate_frame_file_table(
            self.feature_root
        )

    def _load_frame_anno_table(self):
        frame_level_annotation = pl.read_csv(
            os.path.join(
                self.data_root,
                "phoenix-2014-multisigner/annotations/automatic/train.alignment",
            ),
            has_header=False,
            separator=" ",
            new_columns=["relative_path", "class_id"],
            dtypes={
                "relative_path": pl.Utf8,
                "class_id": pl.Int64,
            },
        )
        frame_level_annotation = frame_level_annotation.with_columns(
            (
                os.path.abspath(
                    os.path.join(self.data_root, "phoenix-2014-multisigner")
                )
                + "/"
                + pl.col("relative_path")
            ).alias("full_path")
        )
        class_id_text_table = pl.read_csv(
            os.path.join(
                self.data_root,
                "phoenix-2014-multisigner/annotations/automatic/trainingClasses.txt",
            ),
            separator=" ",
            new_columns=["class_label", "class_id"],
        )
        print(class_id_text_table)
        return frame_level_annotation.join(class_id_text_table, on="class_id")

    def _load_or_generate_frame_file_table(self, feature_root: str):
        abs_feature_root = os.path.abspath(feature_root)
        cache_key = f"frame_file_table_{abs_feature_root}_{self.mode}"

        if cache_key in self.cache and not self.force_reload:
            logger.info("Uinsg cached frame file table.")
            return self.cache[cache_key]
        else:
            if self.force_reload:
                logger.info("Forcing reload of frame file table.")
            logger.info("No cached frame file table found, generating new one.")
            frame_file_table = self._generate_frame_file_table()
            if self.mode == "train":
                frame_file_table = frame_file_table.join(
                    self.frame_anno_table,
                    left_on="frame_file",
                    right_on="full_path",
                    how="left",
                )
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
                        # 01April_2010_Thursday_heute.avi_pid0_fn000002-0.png
                        "frame_index": int(frame_file[-12:-6]),
                    }
                )
        return batch_data

    def _get_video_frame_file_name(self, id: str):
        dir_path = os.path.join(self.feature_root, id, "1")
        logger.debug(f"Looking for frame files in: {dir_path}")
        with os.scandir(dir_path) as it:
            return [
                os.path.abspath(os.path.join(dir_path, entry.name))
                for entry in it
                if entry.name.endswith(".png") and entry.is_file()
            ]

    def get_data_info_by_id(self, id: str):
        if id not in self.ids:
            raise ValueError(f"ID {id} not found in the dataset.")
        data_info = self.raw_annotation.filter(pl.col("id") == id).to_dicts()[0]

        selected = self.frame_file_table.filter(pl.col("id") == id)
        selected = selected.sort("frame_index")

        framefiles = selected["frame_file"].to_list()

        if self.mode == "train":
            # For training mode, we also include the class label
            data_info["class_label"] = selected["class_label"].to_list()

        if not framefiles:
            raise ValueError(f"No frame files found for ID {id}.")
        data_info["frame_files"] = framefiles
        return data_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_root = "dataset/phoenix2014-release"
    ph14t_index = Ph14Index(data_root, "train")
    print(ph14t_index.frame_file_table)
    # print(ph14t_index.frame_file_table["frame_file"][0])
    #
    # print(ph14t_index.frame_anno_table["full_path"][0])
    for ph14t_id in ph14t_index.ids[:10]:
        data_info = ph14t_index.get_data_info_by_id(ph14t_id)
        data_info.pop("frame_files", None)  # Remove frame_files for cleaner output
        print(data_info)
