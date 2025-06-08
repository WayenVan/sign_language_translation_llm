import numpy as np

from nlpaug.augmenter.word.random import RandomWordAug
import polars as pl
import logging

logger = logging.getLogger(__name__)


class RandomWordAugmentation:
    """
    Random Crop Resize for video, not that the parameters used inside the video is the same  across the batch
    """

    def __init__(self, action, *args, **kwargs):
        self.aug = RandomWordAug(action, *args, **kwargs)

    def __call__(self, data: dict) -> dict:
        text = data["text"]
        text = self.aug.augment(text)
        data["text"] = text
        return data


class ExtendedPh14TTextAugmentation:
    """
    Extended PH14T Text Augmentation with extended text data by DeekSeek API.
    """

    def __init__(self, extend_csv_dir):
        self.extend_csv_dir = extend_csv_dir
        self.extended_df = pl.read_csv(
            self.extend_csv_dir,
            separator="|",
        ).select(["id", "text"])

    def __call__(self, data: dict) -> dict:
        original_text = data["text"]

        extended_texts = self.extended_df.filter(self.extended_df["id"] == data["id"])[
            "text"
        ].to_list()

        if len(extended_texts) == 0:
            logger.warning(
                f"Extended text for id {data['id']} not found in {self.extend_csv_dir}"
            )
            selected_text = original_text
        else:
            selected_text = np.random.choice(extended_texts)

        data["text"] = selected_text
        data["original_text"] = original_text
        data["extended_texts"] = extended_texts
        return data
