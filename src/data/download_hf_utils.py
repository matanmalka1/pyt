import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    import sys; sys.exit("[error] Run: pip install datasets")

try:
    from tqdm import tqdm
except ImportError:
    import sys; sys.exit("[error] Run: pip install tqdm")


HF_DATASET    = "hannansatopay/plantvillage-dataset"
CACHE_DIR     = Path(".hf_cache")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="PlantVillage via Hugging Face")
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--cache",    type=str, default=str(CACHE_DIR))
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():           return torch.device("cuda")
    if torch.backends.mps.is_available():   return torch.device("mps")
    return torch.device("cpu")


def get_transforms(split: str, img_size: int) -> transforms.Compose:
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])


class PlantVillageHFDataset(Dataset):
    """
    Wraps a Hugging Face Dataset split as a standard torch.utils.data.Dataset.
    Converts HF PIL images on-the-fly with torchvision transforms.
    """

    def __init__(self, hf_split, transform, label2idx: dict):
        self.data      = hf_split
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        label_raw = sample["label"]
        if isinstance(label_raw, int):
            label = label_raw
        else:
            label = self.label2idx[str(label_raw)]

        return self.transform(img), label


def download_and_prepare(cache_dir: str):
    """
    Stream the PlantVillage dataset from Hugging Face.
    Returns the raw HF DatasetDict and the class list.
    """
    print(f"[hf]  Loading '{HF_DATASET}' from Hugging Face …")
    print(f"[hf]  Cache  → {Path(cache_dir).resolve()}\n")

    hf = load_dataset(HF_DATASET, cache_dir=cache_dir)

    label_feature = hf["train"].features.get("label")
    if hasattr(label_feature, "names"):
        class_names = label_feature.names
    else:
        unique = sorted(set(str(x) for x in hf["train"]["label"]))
        class_names = unique

    return hf, class_names


def build_loaders(hf, class_names, img_size, batch_size, workers):
    label2idx = {name: i for i, name in enumerate(class_names)}

    available = list(hf.keys())
    print(f"[hf]  Available splits: {available}")

    if "validation" not in available and "val" not in available:
        print("[hf]  No validation split found — splitting 90/10 from train …")
        split = hf["train"].train_test_split(test_size=0.1, seed=42)
        hf_train = split["train"]
        hf_val   = split["test"]
    else:
        hf_train = hf["train"]
        hf_val   = hf.get("validation", hf.get("val"))

    hf_test = hf.get("test", hf_val)

    splits = {"train": hf_train, "val": hf_val, "test": hf_test}
    loaders = {}

    for name, hf_split in splits.items():
        ds = PlantVillageHFDataset(hf_split, get_transforms(name, img_size), label2idx)
        loaders[name] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (name == "train"),
            num_workers = workers,
            pin_memory  = True,
        )
        print(f\"[data] {name:<6} → {len(ds):>6,} images\")

    return loaders

