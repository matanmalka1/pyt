# 🌿 PlantVillage Disease Classifier

A production-grade, autonomous PyTorch training pipeline for the
[PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
— classifying 38 crop disease categories with fine-tuned ResNet.

---

## 📁 Project Architecture

```
plantvillage/
│
├── src/                         # All Python source modules
│   ├── train.py                 # ← Main entry point (CLI)
│   ├── data_pipeline.py         # Download, split, transforms, DataLoaders
│   ├── model.py                 # ResNet18/34/50 factory with custom head
│   ├── engine.py                # train/eval epoch loop + predict_batch()
│   ├── predict.py               # Single-image inference script
│   └── utils.py                 # Checkpoint save/load, plotting, reporting
│
├── configs/
│   └── default.yaml             # All hyperparameters in one place
│
├── data/                        # Auto-created at runtime
│   ├── _raw/                    # Extracted archive (temporary)
│   └── plantvillage/
│       ├── train/               # 80% — class sub-folders
│       ├── val/                 # 10% — class sub-folders
│       └── test/                # 10% — class sub-folders
│
├── outputs/                     # Auto-created at runtime
│   ├── best.pth                 # Best checkpoint (by val accuracy)
│   ├── last.pth                 # Most recent epoch checkpoint
│   ├── class_map.json           # {index: class_name} mapping
│   └── training_curve.png       # Loss & accuracy plot
│
├── requirements.txt
├── Makefile
└── README.md
```

---

## ⚙️ Module Responsibilities

| Module | Role |
|--------|------|
| `train.py` | CLI arg parsing, orchestration, training loop, test eval |
| `data_pipeline.py` | Kaggle API download, zip extraction, 80/10/10 split, ImageFolder loaders |
| `model.py` | Pretrained ResNet factory; replaces `fc` layer with `Dropout → Linear(n_classes)` |
| `engine.py` | `run_epoch()` for train + eval; gradient clipping; inline progress bar |
| `utils.py` | `save/load_checkpoint()`, `plot_history()`, `print_summary()`, `save_class_map()` |
| `predict.py` | Single-image inference with top-k output |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd plantvillage

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Kaggle API (Recommended)

```bash
# Place your kaggle.json at:
#   Linux/Mac: ~/.kaggle/kaggle.json
#   Windows:   C:\Users\<user>\.kaggle\kaggle.json

chmod 600 ~/.kaggle/kaggle.json
```

Or manually download the zip and place it in the working directory:
```
plantvillage/plantdisease.zip
```

### 3. Train

```bash
# Default run (ResNet18, 10 epochs)
cd src && python train.py

# Full CLI options
python train.py \
  --backbone  resnet18 \    # resnet18 | resnet34 | resnet50
  --epochs    10        \
  --batch     32        \
  --lr        0.001     \
  --img-size  224       \
  --workers   4         \
  --output    outputs   \
  --no-augment              # disable training augmentation

# Resume from checkpoint
python train.py --resume ../outputs/last.pth --epochs 20
```

### 4. Predict

```bash
python predict.py \
  --image      path/to/leaf.jpg \
  --checkpoint ../outputs/best.pth \
  --class-map  ../outputs/class_map.json \
  --top-k      5
```

---

## 🛠️ Make Commands

```bash
make install          # Install all dependencies
make train            # Train with defaults
make train-fast       # 3-epoch smoke test
make train-resnet50   # Train with ResNet50
make resume           # Resume from last checkpoint
make predict IMAGE=leaf.jpg   # Run inference
make clean            # Remove outputs & cache
make clean-all        # Remove outputs + data + venv
```

---

## 🧠 Model Architecture

```
ResNet18 (pretrained ImageNet)
│
├── conv1   7×7, 64 filters
├── bn1 + relu + maxpool
├── layer1  [BasicBlock × 2]   64 channels
├── layer2  [BasicBlock × 2]  128 channels
├── layer3  [BasicBlock × 2]  256 channels
├── layer4  [BasicBlock × 2]  512 channels
├── avgpool (global)
└── fc      Dropout(0.3) → Linear(512 → n_classes)
```

Total params (ResNet18): ~11.2M → only ~0.2M in `fc` are new.

---

## 📊 Training Pipeline

```
Kaggle API / local zip
        ↓
  Extraction (data/_raw/)
        ↓
  Auto-split → train / val / test (80/10/10 per class)
        ↓
  ImageFolder + Transforms
        ↓
  DataLoader (shuffle, pin_memory, drop_last)
        ↓
  ResNet18 (pretrained) → replace fc
        ↓
  Adam + CosineAnnealingLR
        ↓
  train_epoch → val_epoch → save best.pth
        ↓
  Test evaluation (best weights)
        ↓
  outputs/best.pth + training_curve.png
```

---

## 🔧 Data Augmentation

| Split | Transforms |
|-------|-----------|
| **Train** | RandomResizedCrop(224) · RandomHFlip · RandomVFlip · RandomRotation(20°) · ColorJitter · Normalize |
| **Val / Test** | Resize(256) · CenterCrop(224) · Normalize |

---

## 💻 Hardware Support

| Device | Auto-detected via |
|--------|-------------------|
| NVIDIA GPU | `torch.cuda.is_available()` |
| Apple Silicon | `torch.backends.mps.is_available()` |
| CPU | Fallback |

---

## 📈 Expected Performance

| Backbone | Params | Epochs | ~Val Acc |
|----------|--------|--------|----------|
| ResNet18 | 11.2M  | 10     | 95–97%   |
| ResNet34 | 21.3M  | 10     | 96–98%   |
| ResNet50 | 25.6M  | 15     | 97–99%   |

*(PlantVillage is a relatively clean dataset — high accuracy is expected)*

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `outputs/best.pth` | Best checkpoint (model + optimizer + epoch) |
| `outputs/last.pth` | Most recent checkpoint |
| `outputs/class_map.json` | `{0: "Apple___Apple_scab", ...}` |
| `outputs/training_curve.png` | Loss & accuracy plots |
