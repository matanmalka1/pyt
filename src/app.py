#!/usr/bin/env python3
"""
app.py — FastAPI web interface for PlantVillage training & inference.

Run:
    cd src && uvicorn app:app --reload --port 8000
Then open: http://localhost:8000
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import build_loaders, download_dataset, auto_split
from engine import run_epoch
from model import build_model
from utils import (
    load_checkpoint,
    plot_history,
    save_checkpoint,
    save_class_map,
)

# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PlantVillage AI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Global training state
_train_state = {
    "running": False,
    "log":     [],
    "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
    "best_acc": 0.0,
    "done":    False,
    "error":   None,
}

# ─────────────────────────────────────────────────────────────────────────────
# [FIX] Model cache — loaded once, reused on every predict call.
# Previously the model was loaded from disk on every POST /predict,
# wasting ~1-2s per request and spamming the log.
_model_cache: dict = {
    "model":      None,
    "class_map":  None,
    "backbone":   None,
    "checkpoint": None,   # mtime of best.pth when last loaded
}
_model_lock = threading.Lock()


def _get_device() -> torch.device:
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _load_model_cached(best_pth: Path, class_map_path: Path, device: torch.device):
    """
    Return (model, class_map) from cache.
    Reloads only when best.pth has been modified (i.e. after a new training run).
    Thread-safe via _model_lock.
    """
    with _model_lock:
        current_mtime = best_pth.stat().st_mtime

        if (
            _model_cache["model"] is not None
            and _model_cache["checkpoint"] == current_mtime
        ):
            # Cache hit — return immediately without touching disk
            return _model_cache["model"], _model_cache["class_map"]

        # Cache miss or checkpoint updated — (re)load
        with open(class_map_path) as f:
            class_map = {int(k): v for k, v in json.load(f).items()}

        n_classes = len(class_map)

        ckpt    = torch.load(best_pth, map_location=device)
        fc_in   = ckpt["model_state_dict"]["fc.1.weight"].shape[1]
        backbone = "resnet18" if fc_in == 512 else "resnet50"

        model = build_model(backbone, n_classes, device)
        load_checkpoint(best_pth, model, device=device)
        model.eval()

        _model_cache["model"]      = model
        _model_cache["class_map"]  = class_map
        _model_cache["backbone"]   = backbone
        _model_cache["checkpoint"] = current_mtime

        return model, class_map


def _invalidate_model_cache():
    """Call after training completes so the next predict reloads the new weights."""
    with _model_lock:
        _model_cache["checkpoint"] = None


# ─────────────────────────────────────────────────────────────────────────────
def _run_training(epochs: int, batch: int, lr: float, backbone: str, workers: int):
    """Blocking training loop — runs in a background thread."""
    state = _train_state
    state["running"] = True
    state["done"]    = False
    state["error"]   = None
    state["log"]     = []
    state["history"] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    state["best_acc"] = 0.0

    def log(msg: str):
        state["log"].append(msg)

    try:
        device = _get_device()
        log(f"[device] {device}")

        log("[data] Loading dataset ...")
        download_dataset()
        auto_split()
        loaders, n_classes, class_names = build_loaders(
            Path("data/plantvillage"), batch, 224, workers, augment=True
        )
        save_class_map(class_names, OUTPUT_DIR / "class_map.json")
        log(f"[data] {n_classes} classes loaded")

        model     = build_model(backbone, n_classes, device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        log(f"[model] {backbone.upper()} ready")

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            log(f"EPOCH:{epoch}/{epochs}")
            t0 = time.time()

            tr_loss, tr_acc = run_epoch(
                model, loaders["train"], criterion, optimizer, device, training=True
            )
            vl_loss, vl_acc = run_epoch(
                model, loaders["val"], criterion, None, device, training=False
            )
            scheduler.step()

            for k, v in zip(
                ("train_loss", "val_loss", "train_acc", "val_acc"),
                (tr_loss, vl_loss, tr_acc, vl_acc),
            ):
                state["history"][k].append(round(v, 4))

            elapsed = time.time() - t0
            is_best = vl_acc > best_acc
            if is_best:
                best_acc = vl_acc
                save_checkpoint(OUTPUT_DIR / "best.pth", model, optimizer, epoch, vl_acc)

            save_checkpoint(OUTPUT_DIR / "last.pth", model, optimizer, epoch, vl_acc)
            state["best_acc"] = best_acc

            log(
                f"RESULT:{epoch}|"
                f"{tr_loss:.4f}|{tr_acc:.4f}|"
                f"{vl_loss:.4f}|{vl_acc:.4f}|"
                f"{elapsed:.1f}|{'1' if is_best else '0'}"
            )

        # Final test eval
        log("[test] Evaluating on test set ...")
        load_checkpoint(OUTPUT_DIR / "best.pth", model, device=device)
        te_loss, te_acc = run_epoch(
            model, loaders["test"], criterion, None, device, training=False
        )
        log(f"TEST:{te_loss:.4f}|{te_acc:.4f}")

        plot_history(state["history"], OUTPUT_DIR / "training_curve.png")
        log("DONE")

    except Exception as e:
        state["error"] = str(e)
        log(f"ERROR:{e}")
    finally:
        state["running"] = False
        state["done"]    = True
        # Invalidate cache so next predict loads the freshly trained weights
        _invalidate_model_cache()


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/train/start")
async def train_start(
    epochs:   int   = Form(5),
    batch:    int   = Form(64),
    lr:       float = Form(1e-3),
    backbone: str   = Form("resnet18"),
    workers:  int   = Form(4),
):
    if _train_state["running"]:
        return JSONResponse({"error": "Training already in progress"}, status_code=409)

    thread = threading.Thread(
        target=_run_training,
        args=(epochs, batch, lr, backbone, workers),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@app.get("/train/stream")
async def train_stream():
    """SSE endpoint — streams log lines to the client."""
    async def event_generator():
        sent = 0
        while True:
            logs = _train_state["log"]
            while sent < len(logs):
                yield f"data: {logs[sent]}\n\n"
                sent += 1
            if _train_state["done"] and sent >= len(logs):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/train/status")
async def train_status():
    return {
        "running":  _train_state["running"],
        "done":     _train_state["done"],
        "best_acc": _train_state["best_acc"],
        "history":  _train_state["history"],
        "error":    _train_state["error"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Form(5)):
    class_map_path = OUTPUT_DIR / "class_map.json"
    best_pth       = OUTPUT_DIR / "best.pth"

    if not class_map_path.exists() or not best_pth.exists():
        return JSONResponse(
            {"error": "No trained model found. Please train first."}, status_code=400
        )

    device = _get_device()

    # [FIX] Use cached model instead of loading from disk every time
    model, class_map = _load_model_cached(best_pth, class_map_path, device)

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    thumb = img.copy()
    thumb.thumbnail((300, 300))
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        top_probs, top_idx = probs.topk(top_k, dim=1)

    predictions = [
        {"rank": i + 1, "label": class_map[idx], "confidence": round(prob * 100, 2)}
        for i, (prob, idx) in enumerate(
            zip(top_probs[0].tolist(), top_idx[0].tolist())
        )
    ]

    return {"predictions": predictions, "image": img_b64}