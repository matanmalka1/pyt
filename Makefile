# ─────────────────────────────────────────────────────────────────────────────
# Makefile — PlantVillage Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

PYTHON  = python
SRC     = src
OUTPUTS = outputs
VENV    = .venv

.PHONY: venv install gui train train-fast train-resnet50 resume predict clean clean-all

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual env created. Activate: source $(VENV)/bin/activate"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ── GUI (עיקרי) ───────────────────────────────────────────────────────────────
gui:
	@echo "→ http://localhost:8000"
	cd $(SRC) && uvicorn api.app:app --reload --port 8000

# ── CLI (ישיר) ────────────────────────────────────────────────────────────────
train:
	cd $(SRC) && $(PYTHON) train.py --epochs 5 --batch 64 --lr 0.001 --backbone resnet18

train-fast:
	cd $(SRC) && $(PYTHON) train.py --epochs 2 --batch 128 --no-augment

train-resnet50:
	cd $(SRC) && $(PYTHON) train.py --epochs 8 --batch 32 --lr 0.0005 --backbone resnet50

resume:
	cd $(SRC) && $(PYTHON) train.py --resume ../$(OUTPUTS)/last.pth --epochs 10

predict:
	@echo "Usage: make predict IMAGE=path/to/leaf.jpg"
	cd $(SRC) && $(PYTHON) predict.py \
		--image $(IMAGE) \
		--checkpoint ../$(OUTPUTS)/best.pth \
		--class-map  ../$(OUTPUTS)/class_map.json

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(OUTPUTS) $(SRC)/__pycache__
	@echo "✓ Cleaned"

clean-all: clean
	rm -rf .hf_cache $(VENV)
	@echo "✓ Full clean"
