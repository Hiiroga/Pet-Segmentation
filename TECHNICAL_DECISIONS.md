# TECHNICAL DECISIONS

> Architectural and engineering decisions made during the project.
> Each entry records *what* was decided, *why*, and any *alternatives* considered.

---

## TD-001 — Model Architecture: Custom U-Net

**Decision:** Implement U-Net entirely from scratch using PyTorch `nn.Module`.

**Rationale:**
- Project specification explicitly forbids pretrained encoders, pretrained weights, and third-party segmentation APIs.
- U-Net is well-suited for binary segmentation with limited data — the skip connections preserve spatial detail lost during downsampling.
- A from-scratch implementation gives full control over channel widths, normalisation, and activation choices.

**Alternatives considered:**
- `segmentation_models_pytorch` library → rejected (uses pretrained encoders).
- `torchvision` FCN / DeepLab → rejected (pretrained and API-based).

**Details:**
- Encoder: 4 down-sampling stages with max-pool (channels: 64 → 128 → 256 → 512).
- Bottleneck: 1024 channels.
- Decoder: 4 up-sampling stages with bilinear up-sampling + skip connections.
- Final layer: 1×1 convolution → single-channel logit map (binary).

---

## TD-002 — Loss Function: BCE + Dice (equal weighting)

**Decision:** Combine `BCEWithLogitsLoss` and Dice Loss with equal weights (0.5 each).

**Rationale:**
- BCE provides stable gradient signal across all pixels.
- Dice Loss directly optimises the overlap metric (Dice coefficient) used for evaluation, reducing the mismatch between training loss and evaluation metric.
- Both losses operate on raw logits (before sigmoid), which is numerically more stable.

**Alternatives considered:**
- BCE alone → ignores class imbalance (background usually dominates).
- Focal Loss → adds complexity; reserved for future experimentation.
- Weighted BCE → requires manual foreground/background ratio calculation per batch.

---

## TD-003 — Optimiser: Adam

**Decision:** Use `torch.optim.Adam` with `lr=0.001` and `weight_decay=0.0001`.

**Rationale:**
- Adam adapts per-parameter learning rates, converging reliably on segmentation tasks within 20 epochs.
- L2 regularisation via `weight_decay` prevents overfitting without additional complexity.

**Alternatives considered:**
- SGD with momentum → slower convergence; typically requires learning-rate scheduling.
- AdamW → functionally equivalent for this project; may be adopted in future iterations.

---

## TD-004 — Input Resolution: 256 × 256

**Decision:** Resize all images and masks to 256 × 256 pixels.

**Rationale:**
- Balances spatial detail and GPU memory usage.
- Allows a batch size of 8 on a mid-range GPU (≥6 GB VRAM).
- Oxford-IIIT Pet images vary widely in resolution; uniform resizing is required regardless.

**Alternatives considered:**
- 512 × 512 → 4× more memory; would require batch size ≤ 2 on many GPUs.
- 128 × 128 → too coarse for fine boundary detail.

---

## TD-005 — Training Duration: 20 Epochs (no early stopping)

**Decision:** Train for exactly 20 epochs; early stopping is disabled.

**Rationale:**
- Fixed per project specification.
- Best checkpoint (lowest validation loss) is saved regardless, ensuring the best model is available for evaluation even if the final epoch is not optimal.

---

## TD-006 — Data Splits: 70 / 15 / 15

**Decision:** Split the dataset into 70% training, 15% validation, 15% test.

**Rationale:**
- Standard split ratios for medium-sized datasets.
- Sufficient validation data for reliable loss/metric monitoring.
- Test set is held out and not used during training or hyperparameter selection.

**Implementation:** Stratified split by breed class to ensure balanced class distribution.

---

## TD-007 — Dataset Source: Kaggle (Oxford-IIIT Pet Dataset)

**Decision:** Download via the Kaggle API using the `kaggle` Python package.

**Rationale:**
- Project specification mandates Kaggle as the sole dataset source.
- Automated download via API is reproducible and does not require manual steps after credential setup.

---

## TD-008 — Reproducibility Strategy

**Decision:** Fix seeds for Python `random`, NumPy, and PyTorch (CPU + CUDA). Set `PYTHONHASHSEED`. Enable `cudnn.deterministic`.

**Trade-off:** Disabling `cudnn.benchmark` (which `benchmark=True` enables) may reduce throughput by ~5–10% on some GPUs. Reproducibility is prioritised over raw speed.

---

## TD-009 — Demo Interface: Gradio

**Decision:** Implement the demo with Gradio (not Streamlit or Flask).

**Rationale:**
- Specified by the project brief.
- Minimal boilerplate for image-in / image-out interfaces.
- Built-in support for sliders (threshold control) and side-by-side image display.

---

## TD-010 — Testing Strategy: Pytest + Playwright

**Decision:** Use `pytest` for unit and integration tests; `playwright` for browser-automation tests of the Gradio interface.

**Rationale:**
- `pytest` is the de-facto standard for Python testing.
- `playwright` supports headless browser automation compatible with Gradio's UI.
- Tests are organised in `tests/` and configured via `pytest.ini`.

