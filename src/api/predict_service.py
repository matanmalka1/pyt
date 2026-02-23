import base64
import json
from io import BytesIO
from pathlib import Path

import torch
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from training.model import build_model
from training.utils import load_checkpoint
from inference.grad_cam import GradCAM

from .config import OUTPUT_DIR
from .device import get_device
from .state import model_cache, model_lock


def _load_model_cached(best_pth: Path, class_map_path: Path, device: torch.device):
    """Return (model, class_map) from cache; reload only when best.pth changes."""
    with model_lock:
        current_mtime = best_pth.stat().st_mtime

        if (
            model_cache["model"] is not None
            and model_cache["checkpoint"] == current_mtime
        ):
            return model_cache["model"], model_cache["class_map"]

        with open(class_map_path) as f:
            class_map = {int(k): v for k, v in json.load(f).items()}

        n_classes = len(class_map)
        ckpt      = torch.load(best_pth, map_location=device)
        fc_in     = ckpt["model_state_dict"]["fc.1.weight"].shape[1]
        backbone  = "resnet18" if fc_in == 512 else "resnet50"

        model = build_model(backbone, n_classes, device)
        load_checkpoint(best_pth, model, device=device)
        model.eval()

        model_cache.update({
            "model":      model,
            "class_map":  class_map,
            "backbone":   backbone,
            "checkpoint": current_mtime,
        })
        return model, class_map


def _thumbnail_b64(img: Image.Image) -> str:
    thumb = img.copy()
    thumb.thumbnail((300, 300))
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _gradcam_b64(img: Image.Image, tf, top_idx, class_map):
    gradcam_b64 = None
    try:
        cam_device = torch.device("cpu")
        cam_model  = build_model(model_cache["backbone"], len(class_map), cam_device)
        cam_model.load_state_dict(model_cache["model"].state_dict())
        cam_model.eval()

        cam        = GradCAM(cam_model)
        cam_tensor = tf(img).unsqueeze(0).to(cam_device)
        target_cls = top_idx[0][0].item()

        gradcam_b64 = cam.generate(cam_tensor, target_cls, img)
        cam.remove_hooks()
        del cam_model, cam

    except Exception as e:
        print(f"[gradcam] warning: {e}")

    return gradcam_b64


async def handle_predict(file, top_k: int):
    class_map_path = OUTPUT_DIR / "class_map.json"
    best_pth       = OUTPUT_DIR / "best.pth"

    if not class_map_path.exists() or not best_pth.exists():
        return JSONResponse(
            {"error": "No trained model found. Please train first."}, status_code=400
        )

    device = get_device()
    model, class_map = _load_model_cached(best_pth, class_map_path, device)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    contents = await file.read()
    img      = Image.open(BytesIO(contents)).convert("RGB")

    img_b64 = _thumbnail_b64(img)

    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs              = torch.softmax(model(tensor), dim=1)
        top_probs, top_idx = probs.topk(top_k, dim=1)

    predictions = [
        {"rank": i + 1, "label": class_map[idx], "confidence": round(prob * 100, 2)}
        for i, (prob, idx) in enumerate(zip(top_probs[0].tolist(), top_idx[0].tolist()))
    ]

    gradcam = _gradcam_b64(img, tf, top_idx, class_map)

    return {
        "predictions": predictions,
        "image":       img_b64,
        "gradcam":     gradcam,
    }
