"""
grad_cam.py — Grad-CAM heatmap generation for ResNet models.

Usage:
    from grad_cam import GradCAM
    cam = GradCAM(model)
    heatmap_b64 = cam.generate(tensor, class_idx)
    cam.remove_hooks()
"""

import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet.

    Registers forward/backward hooks on layer4 (the last convolutional block),
    captures activations and gradients, and produces a heatmap overlaid on the
    original image.
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._hooks:       list = []
        self._register_hooks()

    # ── Hook registration ────────────────────────────────────────────────────
    def _register_hooks(self):
        target_layer = self.model.layer4  # last ResNet conv block

        def forward_hook(_, __, output):
            self._activations = output.detach()

        def backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Call when done to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Core computation ─────────────────────────────────────────────────────
    def generate(
        self,
        input_tensor: torch.Tensor,   # (1, 3, H, W) on model device
        class_idx: int,
        original_img: Image.Image,    # PIL image for overlay
        alpha: float = 0.5,           # heatmap opacity
    ) -> str:
        """
        Run a forward+backward pass, compute Grad-CAM, and return
        the heatmap overlaid on original_img as a base64 JPEG string.
        """
        self.model.eval()

        # Forward pass with grad enabled
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        # Backward pass for the target class only
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool gradients over spatial dims → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return self._overlay(cam, original_img, alpha)

    # ── Visualisation ────────────────────────────────────────────────────────
    @staticmethod
    def _overlay(cam: np.ndarray, img: Image.Image, alpha: float) -> str:
        """
        Resize CAM to match img, apply a jet colormap, blend, return base64.
        """
        h, w = img.size[1], img.size[0]

        # Resize CAM to original image size
        cam_pil  = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
        cam_arr  = np.array(cam_pil, dtype=np.float32) / 255.0

        # Jet colormap: blue(cold) → green → red(hot)
        r = np.clip(1.5 - np.abs(cam_arr * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(cam_arr * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(cam_arr * 4 - 1), 0, 1)
        heatmap = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap).convert("RGB")

        # Blend heatmap over original
        orig_rgb = img.convert("RGB")
        blended  = Image.blend(orig_rgb, heatmap_pil, alpha=alpha)

        # Encode to base64
        buf = BytesIO()
        blended.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()
