"""
src/metrics/segmentation_metrics.py
====================================
Epoch-level and batch-level segmentation metrics for binary masks.

Metrics implemented
-------------------
- Dice coefficient  (F1 over pixels)
- IoU / Jaccard index
- Pixel accuracy
- Precision
- Recall
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Epoch-level accumulator
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """
    Accumulates TP / FP / FN / TN across batches, then computes epoch metrics.

    Parameters
    ----------
    threshold : float
        Probability threshold applied after sigmoid to binarise predictions.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self._tp: float = 0.0
        self._fp: float = 0.0
        self._fn: float = 0.0
        self._tn: float = 0.0

    # ------------------------------------------------------------------
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate statistics from one batch.

        Parameters
        ----------
        logits  : [B, 1, H, W]  raw logits (no sigmoid)
        targets : [B, 1, H, W]  binary float (0.0 or 1.0)
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()

            tp = (preds * targets).sum().item()
            fp = (preds * (1.0 - targets)).sum().item()
            fn = ((1.0 - preds) * targets).sum().item()
            tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

            self._tp += tp
            self._fp += fp
            self._fn += fn
            self._tn += tn

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        """
        Compute epoch-level metrics from accumulated statistics.

        Returns
        -------
        dict with keys: dice, iou, pixel_accuracy, precision, recall
        """
        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        eps = 1e-8

        dice           = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou            = (tp + eps) / (tp + fp + fn + eps)
        pixel_accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
        precision      = (tp + eps) / (tp + fp + eps)
        recall         = (tp + eps) / (tp + fn + eps)

        return {
            "dice":            float(dice),
            "iou":             float(iou),
            "pixel_accuracy":  float(pixel_accuracy),
            "precision":       float(precision),
            "recall":          float(recall),
        }


# ---------------------------------------------------------------------------
# Fast single-batch helper
# ---------------------------------------------------------------------------

def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute segmentation metrics for a single batch (no state accumulation).

    Parameters
    ----------
    logits    : [B, 1, H, W]  raw logits
    targets   : [B, 1, H, W]  binary float
    threshold : float

    Returns
    -------
    dict with keys: dice, iou, pixel_accuracy, precision, recall
    """
    m = SegmentationMetrics(threshold=threshold)
    m.update(logits, targets)
    return m.compute()
