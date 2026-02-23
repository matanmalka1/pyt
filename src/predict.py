#!/usr/bin/env python3
"""
predict.py — Run inference on a single image using a trained checkpoint.

Usage:
    python predict.py --image leaf.jpg --checkpoint outputs/best.pth \
                      --class-map outputs/class_map.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from utils import load_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="PlantVillage Inference")
    p.add_argument("--image",      required=True, help="Path to input image")
    p.add_argument("--checkpoint", default="outputs/best.pth")
    p.add_argument("--class-map",  default="outputs/class_map.json")
    p.add_argument("--backbone",   default="resnet18")
    p.add_argument("--top-k",      type=int, default=5)
    p.add_argument("--img-size",   type=int, default=224)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_class_map(path: Path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def preprocess(image_path: str, img_size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0)   # add batch dim


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    device    = get_device()
    class_map = load_class_map(args.class_map)
    n_classes = len(class_map)

    model = build_model(args.backbone, n_classes, device)
    load_checkpoint(Path(args.checkpoint), model, device=device)
    model.eval()

    tensor = preprocess(args.image, args.img_size).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(args.top_k, dim=1)

    print(f"\n  Image     : {args.image}")
    print(f"  Top-{args.top_k} predictions:")
    print(f"  {'Rank':<6} {'Class':<45} {'Confidence':>10}")
    print(f"  {'─'*6} {'─'*45} {'─'*10}")
    for rank, (prob, idx) in enumerate(
        zip(top_probs[0].tolist(), top_indices[0].tolist()), start=1
    ):
        label = class_map.get(idx, f"class_{idx}")
        print(f"  {rank:<6} {label:<45} {prob*100:>9.2f}%")
    print()


if __name__ == "__main__":
    main()
