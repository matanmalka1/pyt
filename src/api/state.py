import threading

# Global training state (mutated by training thread + routes)
train_state = {
    "running": False,
    "log":     [],
    "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
    "best_acc": 0.0,
    "done":    False,
    "error":   None,
}


def reset_train_state():
    train_state.update(
        running=False,
        log=[],
        history={"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
        best_acc=0.0,
        done=False,
        error=None,
    )


# Model cache for prediction endpoint
model_cache: dict = {
    "model":      None,
    "class_map":  None,
    "backbone":   None,
    "checkpoint": None,
}
model_lock = threading.Lock()
