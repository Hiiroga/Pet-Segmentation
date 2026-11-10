# Pet Image Semantic Segmentation

> Binary semantic segmentation of cats and dogs using a custom U-Net implemented from scratch with PyTorch.

---
## Overview

This project implements a complete pixel-level segmentation pipeline for the Oxford-IIIT Pet Dataset. Every component — from the U-Net architecture to the combined loss function — is written from scratch without pretrained encoders, pretrained weights, or third-party segmentation APIs.

| Property        | Value                              |
|-----------------|------------------------------------|
| Task            | Binary semantic segmentation       |
| Model           | Custom U-Net (from scratch)        |
| Framework       | PyTorch                            |
| Dataset         | Oxford-IIIT Pet (via Kaggle)       |
| Input size      | 256 × 256 RGB                      |
| Output          | Binary mask (foreground / background) |
| Loss            | BCEWithLogitsLoss + Dice Loss      |
| Optimizer       | Adam (lr = 0.001, wd = 0.0001)     |
| Epochs          | 20                                 |
| Seed            | 42                                 |

---

## Project Structure

```
pet-segmentation/
├── configs/
│   └── config.yaml            ← Central configuration
├── data/
│   ├── raw/                   ← Downloaded dataset (git-ignored)
│   ├── processed/             ← Resized / normalised files
│   └── splits/                ← Train / val / test CSV splits
├── src/
│   ├── data/                  ← Dataset class, transforms, DataLoaders
│   ├── models/                ← U-Net implementation
│   ├── losses/                ← BCE + Dice combined loss
│   ├── metrics/               ← IoU, Dice, pixel-accuracy
│   ├── training/              ← Training & validation loops
│   ├── inference/             ← Inference pipeline
│   └── utils/                 ← Seed, device, logging, config, dirs
├── checkpoints/               ← Saved model weights
├── outputs/                   ← Visualisations, logs, evaluation
├── tests/                     ← Pytest unit + integration tests
├── app/                       ← Gradio demo (Prompt 07)
├── scripts/                   ← Helper scripts
├── requirements.txt
└── pytest.ini
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Configure Kaggle credentials
Place your `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows).

### 3 — Run project setup
```bash
python scripts/setup_project.py
```

### 4 — Download dataset *(Prompt 02)*
```bash
python scripts/download_dataset.py
```

### 5 — Preprocess data *(Prompt 03)*
```bash
python scripts/preprocess.py
```

### 6 — Train the model *(Prompt 05)*
```bash
python scripts/train.py
```

### 7 — Launch Gradio demo *(Prompt 07)*
```bash
python app/demo.py
```

---

## Running Tests
```bash
pytest
```

---

## Reproducibility

All experiments use seed **42**. Device selection is automatic: CUDA when available, else CPU.

---

## License

For academic and educational use only. Dataset subject to Oxford-IIIT Pet Dataset terms.

