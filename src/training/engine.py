"""
engine.py — Core training and evaluation engine.
Provides run_epoch() used for both training and inference passes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    training:  bool,
) -> Tuple[float, float]:
    """
    Run one full pass over the provided DataLoader.

    Args:
        model     — the neural network
        loader    — DataLoader for the current split
        criterion — loss function (CrossEntropyLoss)
        optimizer — required when training=True, ignored otherwise
        device    — cuda / mps / cpu
        training  — True for training pass, False for eval/inference

    Returns:
        (avg_loss, avg_accuracy)  as floats
    """
    model.train() if training else model.eval()

    running_loss  = 0.0
    correct       = 0
    total         = 0
    n_batches     = len(loader)

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            # FIX: zero_grad must come BEFORE the forward pass, not after backward.
            # Previously it was called after loss.backward(), which wiped out all
            # computed gradients so optimizer.step() always stepped on zeros —
            # the model was never actually learning.
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits  = model(images)
            loss    = criterion(logits, labels)

            if training:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs             = images.size(0)
            running_loss  += loss.item() * bs
            preds          = logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total         += bs

            # Inline progress
            pct = batch_idx / n_batches
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            print(
                f"\r    [{bar}] {batch_idx}/{n_batches}  "
                f"loss={running_loss/total:.4f}  "
                f"acc={correct/total:.3f}",
                end="",
                flush=True,
            )

    print()  # newline after progress bar
    avg_loss = running_loss / total
    avg_acc  = correct / total
    return avg_loss, avg_acc


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_batch(
    model:  nn.Module,
    images: torch.Tensor,
    device: torch.device,
    top_k:  int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on a single batch and return top-k predictions.

    Returns:
        probs  — (B, top_k) softmax probabilities
        indices — (B, top_k) class indices
    """
    model.eval()
    logits     = model(images.to(device))
    probs      = torch.softmax(logits, dim=1)
    return probs.topk(top_k, dim=1)