"""
data_pipeline.py — Automated data acquisition, splitting, transforms, and DataLoaders.
"""

import os
import sys
import shutil
import random
import zipfile
from pathlib import Path
from typing import Tuple, Dict, List

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
DATASET_ID   = "emmarex/plantdisease"
ZIP_FALLBACK = "plantdisease.zip"
SPLITS       = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED         = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
def download_dataset(data_root: Path) -> None:
    """
    Download the PlantVillage dataset via the Kaggle API.
    Falls back to a local zip file if the API is not configured.
    Skips download entirely if data already exists.
    """
    if data_root.exists() and any(data_root.iterdir()):
        print(f"[data] Dataset already present at '{data_root}'. Skipping download.")
        return

    # Try Kaggle API first
    try:
        import kaggle
        kaggle.api.authenticate()
        print(f"[data] Downloading '{DATASET_ID}' from Kaggle …")
        kaggle.api.dataset_download_files(DATASET_ID, path=".", unzip=False)
        print("[data] Download complete.")
    except Exception as exc:
        print(
            f"[data] Kaggle API unavailable: {exc}\n"
            f"       Falling back to local zip: '{ZIP_FALLBACK}'"
        )

    # Locate zip
    zip_path = next(Path(".").glob("*.zip"), None)
    if zip_path is None:
        sys.exit(
            f"[error] No zip file found. "
            f"Download the dataset manually and place '{ZIP_FALLBACK}' "
            f"in the working directory."
        )

    print(f"[data] Extracting '{zip_path}' …")
    extract_dir = Path("data/_raw")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("[data] Extraction complete.")


def _find_class_root(raw: Path) -> Path:
    """
    Recursively locate the directory that directly contains class sub-folders
    (i.e. the folder whose children are the disease categories).
    """
    for depth in range(5):
        candidates = [
            p for p in raw.rglob("*")
            if p.is_dir() and p.stat().st_size == 0 or True  # walk all dirs
        ]
        for p in sorted(raw.rglob("*")):
            if p.is_dir():
                subdirs = [c for c in p.iterdir() if c.is_dir()]
                if len(subdirs) > 2:
                    return p
    return raw


def auto_split(data_root: Path) -> None:
    """
    Split raw images into train / val / test sub-folders if they don't exist.
    Proportions are defined by SPLITS (80/10/10).
    """
    expected = [data_root / s for s in SPLITS]
    if all(d.exists() for d in expected):
        print("[data] Split directories already exist. Skipping split.")
        return

    raw        = Path("data/_raw")
    class_root = _find_class_root(raw)
    classes    = [c for c in class_root.iterdir() if c.is_dir()]

    if not classes:
        sys.exit(
            "[error] Could not locate class directories inside the extracted archive. "
            "Please check the zip structure."
        )

    print(f"[data] Splitting {len(classes)} classes → train/val/test …")
    for cls_dir in classes:
        images = sorted(cls_dir.glob("*.*"))
        random.shuffle(images)
        n      = len(images)
        cut1   = int(n * SPLITS["train"])
        cut2   = int(n * (SPLITS["train"] + SPLITS["val"]))
        parts  = {
            "train": images[:cut1],
            "val":   images[cut1:cut2],
            "test":  images[cut2:],
        }
        for split, files in parts.items():
            dest = data_root / split / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

    print(
        "[data] Split complete — "
        + " | ".join(
            f"{s}: {sum(1 for _ in (data_root/s).rglob('*.*'))} images"
            for s in SPLITS
        )
    )


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
    Build train / val / test DataLoaders from ImageFolder.

    Returns:
        loaders     — dict of DataLoaders
        n_classes   — number of disease classes
        class_names — list of class name strings
    """
    loaders: Dict[str, DataLoader] = {}
    class_names: List[str] = []
    n_classes: int = 0

    for split in ("train", "val", "test"):
        ds = datasets.ImageFolder(
            root      = str(data_root / split),
            transform = get_transforms(split, img_size, augment),
        )
        if not class_names:
            class_names = ds.classes
            n_classes   = len(class_names)

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
        + " | ".join(
            f"{s}: {len(loaders[s].dataset):,} images" for s in loaders
        )
    )
    return loaders, n_classes, class_names
