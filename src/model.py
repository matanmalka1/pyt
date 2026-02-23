"""
model.py — Model factory with flexible backbone selection and fine-tuning options.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

# ─────────────────────────────────────────────────────────────────────────────
_BACKBONE_MAP = {
    "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT),
}


def build_model(
    backbone: str,
    n_classes: int,
    device: torch.device,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Instantiate a pretrained ResNet and replace the classifier head.

    Args:
        backbone        — one of 'resnet18', 'resnet34', 'resnet50'
        n_classes       — number of output disease classes
        device          — target device
        freeze_backbone — if True, freeze all layers except `fc`

    Returns:
        model on the requested device
    """
    if backbone not in _BACKBONE_MAP:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from {list(_BACKBONE_MAP.keys())}"
        )

    factory, weights = _BACKBONE_MAP[backbone]
    model = factory(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print(f"[model] Backbone frozen — only `fc` will be trained.")

    # Replace the final fully-connected layer
    in_features   = model.fc.in_features
    model.fc      = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes),
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] {backbone.upper()}  |  "
        f"{n_classes} classes  |  "
        f"params: {total_params:,} total, {trainable_params:,} trainable"
    )

    return model.to(device)


def model_summary(model: nn.Module, input_shape=(1, 3, 224, 224)) -> None:
    """Print a compact summary of the model layers and output shapes."""
    print("\n── Model Summary ──────────────────────────────────")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {str(module.__class__.__name__):<20} params={params:>10,}")
    print("───────────────────────────────────────────────────\n")
