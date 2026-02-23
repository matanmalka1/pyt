#!/usr/bin/env python3
"""
download_plantvillage_hf.py
────────────────────────────
Downloads PlantVillage from Hugging Face and wraps it as a
standard PyTorch DataLoader

Usage:
    python download_plantvillage_hf.py
    python download_plantvillage_hf.py --batch 64 --img-size 224

Requirements:
    pip install datasets Pillow torch torchvision tqdm
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from download_hf_utils import (  # noqa: E402
    build_loaders,
    download_and_prepare,
    get_device,
    parse_args,
    HF_DATASET,
)


def main():
    args   = parse_args()
    device = get_device()
    print(f"[device] {device}\n")

    hf, class_names = download_and_prepare(args.cache)
    n_classes = len(class_names)
    print(f"\n[data]  {n_classes} classes detected")

    print()
    loaders = build_loaders(hf, class_names, args.img_size, args.batch, args.workers)

    loader = loaders["train"]
    images, labels = next(iter(loader))
    images = images.to(device)

    print(f"\n{'─'*50}")
    print(f"  Batch tensor : {tuple(images.shape)}  dtype={images.dtype}")
    print(f"  Label sample : {labels[:8].tolist()}")
    print(f"  Class sample : {[class_names[i] for i in labels[:4].tolist()]}")
    print(f"  Device       : {device}")
    print(f"{'─'*50}\n")

    print(f"{'═'*50}")
    print(f"  Dataset      : {HF_DATASET}")
    print(f"  Classes      : {n_classes}")
    print(f"  Image size   : {args.img_size}×{args.img_size}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val   batches: {len(loaders['val'])}")
    print(f"{'═'*50}")
    print("\n  ✓ loaders['train'], loaders['val'], loaders['test'] are ready.")
    print("    Pass them directly into your training loop.\n")

    return loaders, class_names, n_classes


if __name__ == "__main__":
    main()
