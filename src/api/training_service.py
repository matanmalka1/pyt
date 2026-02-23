import threading
import time
from pathlib import Path

import torch

from data.data_pipeline import build_loaders, download_dataset, auto_split
from training.engine import run_epoch
from training.model import build_model
from training.utils import load_checkpoint, plot_history, save_checkpoint, save_class_map

from .config import OUTPUT_DIR
from .device import get_device
from .state import train_state
from .state import model_cache, model_lock  # to invalidate after training


def _invalidate_model_cache():
    with model_lock:
        model_cache["checkpoint"] = None


def _run_training(epochs: int, batch: int, lr: float, backbone: str, workers: int):
    state = train_state
    state.update(
        running=True,
        done=False,
        error=None,
        log=[],
        history={"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
        best_acc=0.0,
    )

    def log(msg: str):
        state["log"].append(msg)

    try:
        device = get_device()
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
        _invalidate_model_cache()


def start_background_training(epochs: int, batch: int, lr: float, backbone: str, workers: int):
    if train_state["running"]:
        return False
    thread = threading.Thread(
        target=_run_training,
        args=(epochs, batch, lr, backbone, workers),
        daemon=True,
    )
    thread.start()
    return True
