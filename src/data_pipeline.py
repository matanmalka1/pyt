"""
data_pipeline.py — Data acquisition via Hugging Face datasets, transforms, and DataLoaders.
No Kaggle account or API key required.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    import sys
    sys.exit("[error] Run: pip install datasets")

# ─────────────────────────────────────────────────────────────────────────────
HF_DATASET    = "mohanty/PlantVillage"
HF_CACHE_DIR  = Path(".hf_cache")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
def _detect_columns(hf_train):
    """
    Auto-detect which column contains images and which contains labels
    by inspecting the dataset's feature types.
    Returns (image_col, label_col).
    """
    from datasets import Image as HFImage, ClassLabel, Value

    image_col = None
    label_col = None

    for col, feature in hf_train.features.items():
        if isinstance(feature, HFImage):
            image_col = col
        elif isinstance(feature, ClassLabel):
            label_col = col

    # Fallback: guess by common column name patterns
    if image_col is None:
        for name in hf_train.column_names:
            if "image" in name.lower() or "img" in name.lower():
                image_col = name
                break
    if label_col is None:
        for name in hf_train.column_names:
            if "label" in name.lower() or "class" in name.lower() or "disease" in name.lower():
                label_col = name
                break

    if image_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect image/label columns.\n"
            f"Available columns: {hf_train.column_names}\n"
            f"Features: {hf_train.features}"
        )

    print(f"[data] Detected columns — image: '{image_col}'  label: '{label_col}'")
    return image_col, label_col


# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
def download_dataset(data_root: Path = None) -> None:
    """No-op — data is cached automatically by HuggingFace."""
    print(f"[data] Using HuggingFace '{HF_DATASET}' — no manual download needed.")


def auto_split(data_root: Path = None) -> None:
    """No-op — splits handled inside build_loaders."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str, img_size: int, augment: bool = True) -> transforms.Compose:
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if split == "train" and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])


def build_loaders(
    data_root: Path,
    batch_size: int,
    img_size: int,
    num_workers: int,
    augment: bool = True,
) -> Tuple[Dict[str, DataLoader], int, List[str]]:
    """
    Download (or use cached) PlantVillage from Hugging Face and build
    train / val / test DataLoaders.
    """
    print(f"[data] Loading '{HF_DATASET}' from Hugging Face ...")
    # Load without a named config — works for any layout
    hf = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE_DIR))

    available = list(hf.keys())
    print(f"[data] Available splits: {available}")
    print(f"[data] Columns: {hf['train'].column_names}")

    # Auto-detect which columns are image and label
    image_col, label_col = _detect_columns(hf["train"])

    # Resolve class names
    from datasets import ClassLabel
    label_feature = hf["train"].features.get(label_col)
    if isinstance(label_feature, ClassLabel):
        class_names = label_feature.names
    else:
        class_names = sorted(set(str(x) for x in hf["train"][label_col]))

    label2idx = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    # Build splits — mohanty/PlantVillage has train + test but no val
    if "train" in available and "test" in available and "validation" not in available:
        print("[data] Found train+test — carving 10% of train as val ...")
        tmp      = hf["train"].train_test_split(test_size=0.10, seed=42)
        hf_train = tmp["train"]
        hf_val   = tmp["test"]
        hf_test  = hf["test"]
    elif "validation" not in available and "val" not in available:
        print("[data] Only train split — splitting 80/10/10 ...")
        tmp      = hf["train"].train_test_split(test_size=0.10, seed=42)
        tv       = tmp["train"].train_test_split(test_size=0.10 / 0.90, seed=42)
        hf_train = tv["train"]
        hf_val   = tv["test"]
        hf_test  = tmp["test"]
    else:
        hf_train = hf["train"]
        hf_val   = hf.get("validation", hf.get("val"))
        hf_test  = hf.get("test", hf_val)

    split_map = {"train": hf_train, "val": hf_val, "test": hf_test}
    loaders: Dict[str, DataLoader] = {}

    for split, hf_split in split_map.items():
        ds = PlantVillageDataset(
            hf_split,
            get_transforms(split, img_size, augment),
            label2idx,
            image_col,
            label_col,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = min(num_workers, os.cpu_count() or 1),
            pin_memory  = True,
            drop_last   = (split == "train"),
        )

    print(
        f"[data] {n_classes} classes | "
        + " | ".join(f"{s}: {len(loaders[s].dataset):,} images" for s in loaders)
    )
    return loaders, n_classes, class_names
