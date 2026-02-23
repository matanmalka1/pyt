#!/usr/bin/env python3
"""
train.py — Main entry point for PlantVillage training pipeline.
Orchestrates data loading, model creation, training, and evaluation.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_pipeline import download_dataset, auto_split, build_loaders
from training.model import build_model
from training.engine import run_epoch
from training.utils import save_checkpoint, load_checkpoint, plot_history, print_summary, save_class_map


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="PlantVillage Disease Classifier")
    parser.add_argument("--epochs",    type=int,   default=5,     # was 10; 5 is enough for PlantVillage with transfer learning
                        help="Number of training epochs")
    parser.add_argument("--batch",     type=int,   default=64,    # was 32; larger batch = faster on MPS
                        help="Batch size")
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--img-size",  type=int,   default=224)
    parser.add_argument("--data-root", type=str,   default="data/plantvillage")
    parser.add_argument("--output",    type=str,   default="outputs")
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--backbone",  type=str,   default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Backbone architecture")
    parser.add_argument("--workers",   type=int,   default=4)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training augmentation")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Apple MPS")
    else:
        device = torch.device("cpu")
        print("[device] CPU")
    return device


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    download_dataset(data_root)
    auto_split(data_root)
    loaders, n_classes, class_names = build_loaders(
        data_root, args.batch, args.img_size, args.workers, not args.no_augment
    )

    save_class_map(class_names, output_dir / "class_map.json")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model     = build_model(args.backbone, n_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_acc    = 0.0
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, device)

    # ── 3. Training loop ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PlantVillage | {args.backbone.upper()} | {n_classes} classes")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch}  LR: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}  LR={scheduler.get_last_lr()[0]:.2e}")

        print("  [train]", end=" ")
        tr_loss, tr_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, training=True
        )

        print("  [val]  ", end=" ")
        vl_loss, vl_acc = run_epoch(
            model, loaders["val"], criterion, None, device, training=False
        )

        scheduler.step()

        for k, v in zip(
            ("train_loss", "val_loss", "train_acc", "val_acc"),
            (tr_loss, vl_loss, tr_acc, vl_acc),
        ):
            history[k].append(v)

        elapsed = time.time() - t0
        print(
            f"  → train loss={tr_loss:.4f}  acc={tr_acc:.3f} | "
            f"val loss={vl_loss:.4f}  acc={vl_acc:.3f} | {elapsed:.1f}s"
        )

        is_best = vl_acc > best_acc
        if is_best:
            best_acc = vl_acc

        save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, vl_acc)
        if is_best:
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, vl_acc)
            print(f"  ✓ Best model saved  (val_acc={best_acc:.4f})")

    # ── 4. Test evaluation ────────────────────────────────────────────────────
    print("\n[test] Loading best weights …")
    load_checkpoint(output_dir / "best.pth", model, device=device)

    print("[test] Running on held-out test set …")
    te_loss, te_acc = run_epoch(
        model, loaders["test"], criterion, None, device, training=False
    )

    print_summary(history, best_acc, te_loss, te_acc, n_classes, args.backbone)
    plot_history(history, output_dir / "training_curve.png")
    print(f"\n[done] Outputs saved to '{output_dir}/'")


if __name__ == "__main__":
    main()