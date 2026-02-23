"""
utils.py — Checkpoint management, plotting, and result reporting.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_acc:   float,
) -> None:
    """Persist model weights, optimizer state, and training metadata."""
    torch.save(
        {
            "epoch":      epoch,
            "val_acc":    val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """
    Restore model (and optionally optimizer) from a checkpoint.

    Returns:
        (start_epoch, best_val_acc)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch   = ckpt.get("epoch", 0) + 1
    val_acc = ckpt.get("val_acc", 0.0)
    print(f"[ckpt] Loaded '{path}'  (epoch {epoch-1}, val_acc={val_acc:.4f})")
    return epoch, val_acc


# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history: Dict[str, List[float]], save_path: Path) -> None:
    """
    Plot training/validation loss and accuracy curves and save to disk.
    Falls back to a text-based ASCII chart if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless / non-interactive
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("PlantVillage Training Curve", fontsize=14, fontweight="bold")

        # Loss
        ax1.plot(epochs, history["train_loss"], "b-o", label="Train")
        ax1.plot(epochs, history["val_loss"],   "r-o", label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history["train_acc"], "b-o", label="Train")
        ax2.plot(epochs, history["val_acc"],   "r-o", label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved training curves → '{save_path}'")

    except ImportError:
        print("[plot] matplotlib not found — printing ASCII summary instead.")
        _ascii_plot(history)


def _ascii_plot(history: Dict[str, List[float]]) -> None:
    """Minimal ASCII representation of validation accuracy over epochs."""
    accs    = history["val_acc"]
    n       = len(accs)
    print("\n  Val Accuracy over Epochs:")
    for i, acc in enumerate(accs, 1):
        bar = "█" * int(acc * 40)
        print(f"  Epoch {i:>2}  [{bar:<40}] {acc:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(
    history:    Dict[str, List[float]],
    best_acc:   float,
    te_loss:    float,
    te_acc:     float,
    n_classes:  int,
    backbone:   str,
) -> None:
    """Print a formatted results table to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TRAINING COMPLETE — FINAL RESULTS")
    print(sep)
    print(f"  Backbone        : {backbone.upper()}")
    print(f"  Classes         : {n_classes}")
    print(f"  Best Val Acc    : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"  Test  Loss      : {te_loss:.4f}")
    print(f"  Test  Accuracy  : {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(sep)
    print(f"\n  Val Acc per epoch:")
    for i, acc in enumerate(history["val_acc"], 1):
        bar = "█" * int(acc * 30)
        print(f"    Epoch {i:>2}  {bar:<30} {acc:.3f}")
    print()


def save_class_map(class_names: List[str], path: Path) -> None:
    """Persist the class-index mapping as JSON for inference use."""
    mapping = {i: name for i, name in enumerate(class_names)}
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[utils] Class map saved → '{path}'")
