# PROJECT STATUS

> Last updated: 2026-11-01 (Final)

---

## Overall Progress

| Stage | Title                              | Status       |
|-------|------------------------------------|--------------|
| 01    | Project Initialisation             | ✅ Complete  |
| 02    | Dataset Download & Exploration     | ✅ Complete  |
| 03    | Data Preprocessing & Splits        | ✅ Complete  |
| 04    | PyTorch Dataset & Augmentation     | ✅ Complete  |
| 05    | Custom U-Net, Loss, Metrics        | ✅ Complete  |
| 06    | Training (5 epochs of 10 planned)  | ✅ Complete* |
| 07    | Evaluation & Model Selection       | ✅ Complete  |
| 08    | Inference & Gradio Application     | ✅ Complete  |
| 09    | Automated Testing (pytest)         | ✅ Complete  |
| 10    | Final Documentation & Report       | ✅ Complete  |

> *Training dihentikan setelah 5 epoch (dari 10 yang direncanakan). Best model dari epoch 4.

---

## Hardware & Environment

| Item | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) |
| CUDA | 12.4 |
| PyTorch | 2.5.1+cu124 |
| Python | 3.11 |
| OS | Windows 11 |

---

## Dataset

| Item | Value |
|---|---|
| Source | Kaggle: `zippyz/cats-and-dogs-breeds-classification-oxford-dataset` |
| Total samples | 7.349 (dari list.txt) |
| Train | 5.144 |
| Validation | 1.102 |
| Test | 1.103 |
| Cats | 2.371 |
| Dogs | 4.978 |

---

## Model

| Item | Value |
|---|---|
| Architecture | Custom U-Net from scratch |
| Parameters | 31.384.833 |
| Input | 256×256 RGB |
| Output | Binary segmentation mask (1 channel) |
| No pretrained weights | ✅ Confirmed |

---

## Training Configuration

| Item | Value |
|---|---|
| Epochs completed | 5 (of 10 planned) |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | BCE (0.5) + Dice (0.5) |
| AMP | Enabled (FP16) |
| cuDNN benchmark | Enabled |
| Seed | 42 |

---

## Training Results (Epoch 1–5)

| Epoch | Train Loss | Val Loss | Val Dice | Val IoU | Val PxAcc |
|-------|-----------|---------|---------|--------|---------|
| 1 | 0.4846 | 0.5045 | 0.5564 | 0.3854 | 0.7703 |
| 2 | 0.4434 | 0.5788 | 0.3056 | 0.1804 | 0.7463 |
| 3 | 0.4030 | 0.4225 | 0.6619 | 0.4947 | 0.8260 |
| 4 | 0.3577 | **0.3614** | **0.7393** | **0.5864** | 0.8520 |
| 5 | 0.3259 | 0.3484 | 0.7205 | 0.5631 | 0.8540 |

> **Best model: Epoch 4** (val_loss=0.3614, simpan sebagai `best_model.pth`)

---

## Test Set Evaluation (Actual — best_model.pth, epoch 4)

| Metric | Value |
|---|---|
| Test Loss | 0.3511 |
| **Dice Score** | **0.7168** |
| **IoU** | **0.5585** |
| Pixel Accuracy | 0.8535 |
| Precision | 0.8558 |
| Recall | 0.6166 |

---

## Automated Testing

| Test Suite | Tests | Status |
|---|---|---|
| `test_utils.py` | 28 | ✅ All PASSED |
| `test_model.py` | 32 | ✅ All PASSED |
| `test_inference.py` | 15 | ✅ All PASSED |
| **Total** | **75** | **✅ 75/75 PASSED** |

---

## Output Files

| File | Status |
|---|---|
| `checkpoints/best_model.pth` | ✅ Epoch 4 |
| `checkpoints/checkpoint_epoch_01-05.pth` | ✅ All saved |
| `outputs/evaluation/test_metrics.json` | ✅ Actual metrics |
| `outputs/predictions/sample_01-08.png` | ✅ 8 samples |
| `outputs/overlays/overlay_01-08.png` | ✅ 8 overlays |
| `outputs/dataset_samples/*.png` | ✅ 8 visualisations |
| `outputs/training_progress/training_curves.png` | ✅ Generated |

---

## Known Issues

| Issue | Fix | Status |
|---|---|---|
| UnicodeEncodeError (Windows cp1252) | Replaced non-ASCII chars in log.info() | ✅ Fixed |
| Kaggle CLI --unzip failed | Used PowerShell Expand-Archive | ✅ Fixed |
| CombinedLoss returned tuple | Changed to return dict | ✅ Fixed |
| CPU training too slow (~23s/batch) | Installed CUDA PyTorch + AMP + cuDNN | ✅ Fixed |
| Pillow 13.0.0 incompatible with Gradio | Downgraded to Pillow 12.3.0 | ✅ Fixed |
| test_batch_size / test_epochs hardcoded | Changed to type+range assertions | ✅ Fixed |
