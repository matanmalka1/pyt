# ─────────────────────────────────────────────────────────────────────────────
# Makefile — PlantVillage Training Pipeline
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON     = python
SRC        = src
OUTPUTS    = outputs
VENV       = .venv

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: venv install clean

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual env created at '$(VENV)/'"
	@echo "  Activate: source $(VENV)/bin/activate"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ── Training ──────────────────────────────────────────────────────────────────
.PHONY: train train-fast train-resnet50 resume

train:
	cd $(SRC) && $(PYTHON) train.py \
		--epochs 10 --batch 32 --lr 0.001 --backbone resnet18

train-fast:
	cd $(SRC) && $(PYTHON) train.py \
		--epochs 3 --batch 64 --lr 0.001 --backbone resnet18 --no-augment

train-resnet50:
	cd $(SRC) && $(PYTHON) train.py \
		--epochs 15 --batch 16 --lr 0.0005 --backbone resnet50

resume:
	cd $(SRC) && $(PYTHON) train.py \
		--resume ../$(OUTPUTS)/last.pth --epochs 20

# ── Inference ─────────────────────────────────────────────────────────────────
.PHONY: predict

predict:
	@echo "Usage: make predict IMAGE=path/to/leaf.jpg"
	cd $(SRC) && $(PYTHON) predict.py \
		--image $(IMAGE) \
		--checkpoint ../$(OUTPUTS)/best.pth \
		--class-map  ../$(OUTPUTS)/class_map.json

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(OUTPUTS) $(SRC)/__pycache__
	@echo "✓ Cleaned outputs and cache"

clean-all: clean
	rm -rf .hf_cache $(VENV)
	@echo "✓ Full clean complete (HF cache + venv removed)"