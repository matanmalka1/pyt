import os
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from .config import HF_DATASET, HF_CACHE_DIR
from .dataset import PlantVillageDataset, _detect_columns
from .transforms import get_transforms

try:
    from datasets import load_dataset
except ImportError:
    import sys
    sys.exit("[error] Run: pip install datasets")


def download_dataset(data_root=None) -> None:
    """No-op — HuggingFace caches automatically."""
    print(f"[data] Using HuggingFace '{HF_DATASET}' — no manual download needed.")


def auto_split(data_root=None) -> None:
    """No-op — splits handled inside build_loaders."""
    pass


def build_loaders(
    data_root,
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
    hf = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE_DIR))

    available = list(hf.keys())
    print(f"[data] Available splits: {available}")
    print(f"[data] Columns:          {hf['train'].column_names}")

    image_col, label_col = _detect_columns(hf["train"])

    from datasets import ClassLabel
    label_feature = hf["train"].features.get(label_col)
    if isinstance(label_feature, ClassLabel):
        class_names = label_feature.names
    else:
        class_names = sorted(set(str(x) for x in hf["train"][label_col]))

    label2idx = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    if "train" in available and "test" in available and "validation" not in available:
        print("[data] Found train+test — carving 10% of train as val ...")
        tmp      = hf["train"].train_test_split(test_size=0.10, seed=42)
        hf_train = tmp["train"]
        hf_val   = tmp["test"]
        hf_test  = hf["test"]
    elif len(available) == 1:
        print("[data] Only one split — carving 80/10/10 ...")
        tmp      = hf[available[0]].train_test_split(test_size=0.10, seed=42)
        tv       = tmp["train"].train_test_split(test_size=0.10 / 0.90, seed=42)
        hf_train = tv["train"]
        hf_val   = tv["test"]
        hf_test  = tmp["test"]
    else:
        hf_train = hf["train"]
        hf_val   = hf.get("validation", hf.get("val", hf.get("test")))
        hf_test  = hf.get("test", hf_val)

    split_map = {"train": hf_train, "val": hf_val, "test": hf_test}
    loaders: Dict[str, DataLoader] = {}

    effective_workers = min(num_workers, os.cpu_count() or 1)
    use_prefetch      = effective_workers > 0

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
            batch_size         = batch_size,
            shuffle            = (split == "train"),
            num_workers        = effective_workers,
            pin_memory         = False,
            drop_last          = (split == "train"),
            prefetch_factor    = 2 if use_prefetch else None,
            persistent_workers = use_prefetch,
        )

    print(
        f\"[data] {n_classes} classes | "
        + " | ".join(f\"{s}: {len(loaders[s].dataset):,} images\" for s in loaders)
    )
    return loaders, n_classes, class_names
