import asyncio
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import STATIC_DIR
from .predict_service import handle_predict
from .state import train_state
from .training_service import start_background_training

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@router.post("/train/start")
async def train_start(
    epochs:   int   = Form(5),
    batch:    int   = Form(64),
    lr:       float = Form(1e-3),
    backbone: str   = Form("resnet18"),
    workers:  int   = Form(4),
):
    started = start_background_training(epochs, batch, lr, backbone, workers)
    if not started:
        return JSONResponse({"error": "Training already in progress"}, status_code=409)
    return {"status": "started"}


@router.get("/train/stream")
async def train_stream():
    async def event_generator():
        sent = 0
        while True:
            logs = train_state["log"]
            while sent < len(logs):
                yield f"data: {logs[sent]}\n\n"
                sent += 1
            if train_state["done"] and sent >= len(logs):
                break
            await asyncio.sleep(0.3)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/train/status")
async def train_status():
    return {
        "running":  train_state["running"],
        "done":     train_state["done"],
        "best_acc": train_state["best_acc"],
        "history":  train_state["history"],
        "error":    train_state["error"],
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Form(5)):
    return await handle_predict(file, top_k)
