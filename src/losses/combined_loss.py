"""
src/losses/combined_loss.py
===========================
Dice Loss and Combined (BCE + Dice) Loss for binary segmentation.

Both losses expect *raw logits* (before sigmoid) as input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Dice loss computed on raw logits (sigmoid applied internally).

    Parameters
    ----------
    smooth : float
        Additive smoothing constant to avoid division-by-zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, 1, H, W]  raw predictions (no sigmoid)
        targets : [B, 1, H, W]  binary float (0.0 or 1.0)

        Returns
        -------
        torch.Tensor — scalar Dice loss (mean over batch)
        """
        probs = torch.sigmoid(logits)

        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_sample.mean()


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Weighted sum of BCEWithLogitsLoss and DiceLoss.

    forward() returns a dict:
        "total" : torch.Tensor  — scalar, differentiable (.backward() on this)
        "bce"   : torch.Tensor  — BCE component (detached scalar)
        "dice"  : torch.Tensor  — Dice component (detached scalar)
    """

    def __init__(
        self,
        bce_weight:  float = 0.5,
        dice_weight: float = 0.5,
        smooth:      float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce_fn  = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Returns
        -------
        dict:
            "total" : torch.Tensor — differentiable scalar (use for .backward())
            "bce"   : torch.Tensor — BCE component
            "dice"  : torch.Tensor — Dice component
        """
        bce_loss  = self.bce_fn(logits, targets)
        dice_loss = self.dice_fn(logits, targets)
        total     = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return {
            "total": total,
            "bce":   bce_loss,
            "dice":  dice_loss,
        }
