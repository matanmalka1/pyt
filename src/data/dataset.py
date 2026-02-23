import os
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from .config import HF_CACHE_DIR
from .transforms import get_transforms


class PlantVillageDataset(Dataset):
    """
    Wraps a Hugging Face dataset split as a standard torch Dataset.
    Applies torchvision transforms on-the-fly.
    """

    def __init__(self, hf_split, transform, label2idx: dict,
                 image_col: str, label_col: str):
        self.data      = hf_split
        self.transform = transform
        self.label2idx = label2idx
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img = sample[self.image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        label_raw = sample[self.label_col]
        label = label_raw if isinstance(label_raw, int) else self.label2idx[str(label_raw)]

        return self.transform(img), label


def _detect_columns(hf_train):
    """Auto-detect image and label columns from HF feature types."""
    from datasets import Image as HFImage, ClassLabel

    image_col = label_col = None

    for col, feature in hf_train.features.items():
        if isinstance(feature, HFImage):
            image_col = col
        elif isinstance(feature, ClassLabel):
            label_col = col

    if image_col is None:
        for name in hf_train.column_names:
            if "image" in name.lower() or "img" in name.lower():
                image_col = name; break
    if label_col is None:
        for name in hf_train.column_names:
            if "label" in name.lower() or "class" in name.lower():
                label_col = name; break

    if image_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect image/label columns.\n"
            f"Columns: {hf_train.column_names}\nFeatures: {hf_train.features}"
        )

    print(f"[data] Columns → image: '{image_col}'  label: '{label_col}'")
    return image_col, label_col
