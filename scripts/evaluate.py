"""
scripts/evaluate.py
===================
Stage 07 - Evaluation & Model Selection.

Loads best_model.pth, runs inference on the full test split, computes
segmentation metrics, saves sample visualisations, and writes a markdown report.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_20.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.data.dataset import get_dataloaders
from src.models.unet import UNet
from src.losses.combined_loss import CombinedLoss
from src.metrics.segmentation_metrics import SegmentationMetrics

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path, cfg: dict, device: torch.device) -> UNet:
    model_cfg = cfg["model"]
    model = UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        base_features=model_cfg.get("base_features", [64, 128, 256, 512]),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    log.info("Checkpoint loaded (epoch %s): %s", epoch, checkpoint_path)
    return model


def _run_evaluation(model, test_loader, criterion, metrics_acc, device):
    """Iterate the test loader; return (avg_loss, metric_dict, sample_batches)."""
    running_loss = 0.0
    n_batches = len(test_loader)
    sample_batches = []  # store first 3 batches for visualisation

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(
            tqdm(test_loader, desc="Evaluating", unit="batch")
        ):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            logits    = model(images)
            loss_dict = criterion(logits, masks)
            running_loss += loss_dict["total"].item()
            metrics_acc.update(logits, masks)

            if batch_idx < 3:
                sample_batches.append(
                    (images.cpu(), masks.cpu(), logits.cpu())
                )

    avg_loss    = running_loss / max(n_batches, 1)
    metric_dict = metrics_acc.compute()
    return avg_loss, metric_dict, sample_batches


def _save_sample_visualisations(
    sample_batches,
    predictions_dir: Path,
    overlays_dir: Path,
    max_samples: int = 8,
):
    """
    Save side-by-side PNG files (image | GT mask | prediction | overlay)
    for up to max_samples images drawn from the first few test batches.
    """
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    saved = 0
    for images, masks, logits in sample_batches:
        probs  = torch.sigmoid(logits).squeeze(1).numpy()   # (B, H, W)
        binary = (probs >= 0.5).astype(np.uint8)            # (B, H, W) 0/1
        imgs_np = images.numpy()                             # (B, 3, H, W)
        masks_np = masks.squeeze(1).numpy()                  # (B, H, W)

        for i in range(len(images)):
            if saved >= max_samples:
                return

            # Denormalise image for display
            img = imgs_np[i].transpose(1, 2, 0)   # (H, W, 3)
            img = (img * _STD + _MEAN).clip(0, 1)

            gt_mask   = masks_np[i]                # (H, W) float 0/1
            pred_mask = binary[i]                  # (H, W) int 0/1

            # Overlay: green = correct FG, red = FP, yellow = FN
            overlay = (img * 255).astype(np.uint8).copy()
            tp = (pred_mask == 1) & (gt_mask > 0.5)
            fp = (pred_mask == 1) & (gt_mask < 0.5)
            fn = (pred_mask == 0) & (gt_mask > 0.5)
            overlay[tp] = [0, 200, 0]
            overlay[fp] = [200, 0, 0]
            overlay[fn] = [255, 215, 0]

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            axes[3].imshow(overlay)
            axes[3].set_title("Overlay (G=TP R=FP Y=FN)")
            axes[3].axis("off")

            plt.tight_layout()
            pred_path    = predictions_dir / f"sample_{saved+1:02d}.png"
            overlay_path = overlays_dir    / f"overlay_{saved+1:02d}.png"
            plt.savefig(pred_path, dpi=120)
            plt.savefig(overlay_path, dpi=120)
            plt.close(fig)
            log.info("Saved visualisation %d: %s", saved + 1, pred_path)
            saved += 1


def _write_report(
    report_path: Path,
    checkpoint_path: Path,
    avg_loss: float,
    metric_dict: dict,
    n_test: int,
):
    """Write a markdown evaluation report."""
    lines = [
        "# Evaluation Report — Pet Segmentation U-Net",
        "",
        "## Checkpoint",
        f"- **File:** `{checkpoint_path}`",
        "",
        "## Test Set",
        f"- **Samples evaluated:** {n_test}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test Loss | {avg_loss:.4f} |",
        f"| Dice Coefficient | {metric_dict['dice']:.4f} |",
        f"| IoU (Jaccard) | {metric_dict['iou']:.4f} |",
        f"| Pixel Accuracy | {metric_dict['pixel_accuracy']:.4f} |",
        f"| Precision | {metric_dict.get('precision', float('nan')):.4f} |",
        f"| Recall | {metric_dict.get('recall', float('nan')):.4f} |",
        "",
        "## Outputs",
        "- Sample predictions: `outputs/predictions/`",
        "- Overlay visualisations: `outputs/overlays/`",
        "",
        "> All metrics computed on the held-out test split (no leakage from training).",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Evaluation report written -> %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    log.info("Device: %s", device)

    # Resolve checkpoint path
    ckpt_cfg      = cfg["checkpoints"]
    checkpoint_dir = _PROJECT_ROOT / ckpt_cfg["dir"]
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = checkpoint_dir / ckpt_cfg.get("best_model_name", "best_model.pth")

    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        log.error("Train the model first with: python scripts/train.py")
        sys.exit(1)

    # Dirs
    outputs_cfg     = cfg["outputs"]
    predictions_dir = _PROJECT_ROOT / outputs_cfg["predictions"]
    overlays_dir    = _PROJECT_ROOT / outputs_cfg["overlays"]
    evaluation_dir  = _PROJECT_ROOT / outputs_cfg["evaluation"]
    for d in (predictions_dir, overlays_dir, evaluation_dir):
        d.mkdir(parents=True, exist_ok=True)

    # DataLoaders (we only use test_loader)
    _, _, test_loader = get_dataloaders(cfg)
    n_test = len(test_loader.dataset)
    log.info("Test samples: %d  | batches: %d", n_test, len(test_loader))

    # Model
    model = _load_model(checkpoint_path, cfg, device)

    # Loss
    loss_cfg  = cfg["loss"]
    criterion = CombinedLoss(
        bce_weight=loss_cfg.get("bce_weight", 0.5),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        smooth=loss_cfg.get("dice_smooth", 1.0),
    )

    # Metrics accumulator
    threshold   = cfg.get("inference", {}).get("threshold", 0.5)
    metrics_acc = SegmentationMetrics(threshold=threshold)

    # Evaluate
    avg_loss, metric_dict, sample_batches = _run_evaluation(
        model, test_loader, criterion, metrics_acc, device
    )

    log.info("=== Test Results ===")
    log.info("  Loss           : %.4f", avg_loss)
    log.info("  Dice           : %.4f", metric_dict["dice"])
    log.info("  IoU            : %.4f", metric_dict["iou"])
    log.info("  Pixel Accuracy : %.4f", metric_dict["pixel_accuracy"])
    log.info("  Precision      : %.4f", metric_dict.get("precision", float("nan")))
    log.info("  Recall         : %.4f", metric_dict.get("recall", float("nan")))

    # Save JSON metrics
    metrics_out = {
        "checkpoint": str(checkpoint_path),
        "n_test_samples": n_test,
        "test_loss": avg_loss,
        **metric_dict,
    }
    json_path = evaluation_dir / "test_metrics.json"
    json_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log.info("Metrics JSON saved -> %s", json_path)

    # Save sample visualisations
    _save_sample_visualisations(sample_batches, predictions_dir, overlays_dir)

    # Write markdown report
    _write_report(
        evaluation_dir / "evaluation_report.md",
        checkpoint_path,
        avg_loss,
        metric_dict,
        n_test,
    )

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()

