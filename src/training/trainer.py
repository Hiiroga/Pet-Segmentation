"""
src/training/trainer.py
=======================
Complete training loop with epoch-level logging, checkpointing,
and training-history persistence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.metrics.segmentation_metrics import SegmentationMetrics
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the full training and validation loop for a segmentation model.

    Parameters
    ----------
    model          : nn.Module
    optimizer      : torch.optim.Optimizer
    criterion      : callable — must return (loss_tensor, loss_dict)
    device         : torch.device
    cfg            : dict — parsed config
    checkpoint_dir : str | Path
    output_dir     : str | Path — for saving training_history.json and plots
    """

    def __init__(
        self,
        model:          nn.Module,
        optimizer:      torch.optim.Optimizer,
        criterion:      Any,
        device:         torch.device,
        cfg:            dict,
        checkpoint_dir: str | Path,
        output_dir:     str | Path,
    ) -> None:
        self.model          = model
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.device         = device
        self.cfg            = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir     = Path(output_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Automatic Mixed Precision
        use_amp = cfg.get("training", {}).get("amp", False) and device.type == "cuda"
        self._scaler = GradScaler() if use_amp else None
        self._use_amp = use_amp
        if use_amp:
            log.info("Automatic Mixed Precision (AMP) enabled.")

        self._best_val_loss: float = float("inf")
        self._metrics       = SegmentationMetrics(
            threshold=cfg.get("inference", {}).get("threshold", 0.5)
        )

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one training epoch. Returns averaged metric dict."""
        self.model.train()
        self._metrics.reset()
        running_loss = 0.0
        n_batches    = len(loader)

        pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(enabled=self._use_amp):
                logits    = self.model(images)
                loss_dict = self.criterion(logits, masks)
                loss      = loss_dict["total"]

            if self._scaler:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            self._metrics.update(logits.detach(), masks)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        metric_dict = self._metrics.compute()
        return {
            "loss":           avg_loss,
            "dice":           metric_dict["dice"],
            "iou":            metric_dict["iou"],
            "pixel_accuracy": metric_dict["pixel_accuracy"],
        }

    # ------------------------------------------------------------------
    def validate_epoch(self, loader: DataLoader) -> dict:
        """Run one validation epoch (no gradient computation)."""
        self.model.eval()
        self._metrics.reset()
        running_loss = 0.0
        n_batches    = len(loader)

        pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device,  non_blocking=True)

                logits    = self.model(images)
                loss_dict = self.criterion(logits, masks)
                loss      = loss_dict["total"]
                running_loss += loss.item()
                self._metrics.update(logits, masks)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        metric_dict = self._metrics.compute()
        return {
            "loss":           avg_loss,
            "dice":           metric_dict["dice"],
            "iou":            metric_dict["iou"],
            "pixel_accuracy": metric_dict["pixel_accuracy"],
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch:    int,
        metrics:  dict,
        is_best:  bool = False,
    ) -> None:
        """
        Save a checkpoint to disk.

        Files saved
        -----------
        checkpoints/checkpoint_epoch_{epoch:02d}.pth
        checkpoints/best_model.pth  (if is_best=True)
        """
        checkpoint = {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":              metrics,
            "cfg":                  self.cfg,
        }

        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.pth"
        torch.save(checkpoint, ckpt_path)
        log.info("Checkpoint saved -> %s", ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / self.cfg.get(
                "checkpoints", {}
            ).get("best_model_name", "best_model.pth")
            torch.save(checkpoint, best_path)
            log.info("Best model saved -> %s", best_path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str | Path) -> dict:
        """
        Load a checkpoint from disk.

        Returns
        -------
        dict  — full checkpoint dict (epoch, metrics, cfg, …)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        log.info("Checkpoint loaded from %s (epoch %d)", checkpoint_path, checkpoint["epoch"])
        return checkpoint

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        start_epoch:  int = 1,
    ) -> dict:
        """
        Full multi-epoch training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader   : DataLoader
        start_epoch  : int  — epoch to resume from (1-indexed)

        Returns
        -------
        dict — training history with lists of per-epoch metrics
        """
        total_epochs = self.cfg["training"]["epochs"]

        history: dict[str, Any] = {
            "train_loss": [],
            "val_loss":   [],
            "train_dice": [],
            "val_dice":   [],
            "train_iou":  [],
            "val_iou":    [],
            "epochs_completed": 0,
        }

        log.info(
            "Starting training: epochs %d-%d on device=%s",
            start_epoch, total_epochs, self.device,
        )

        for epoch in range(start_epoch, total_epochs + 1):
            log.info("=== Epoch %d / %d ===", epoch, total_epochs)

            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate_epoch(val_loader)

            # Log metrics
            log.info(
                "Train -> loss=%.4f  dice=%.4f  iou=%.4f  px_acc=%.4f",
                train_metrics["loss"], train_metrics["dice"],
                train_metrics["iou"], train_metrics["pixel_accuracy"],
            )
            log.info(
                "Val   -> loss=%.4f  dice=%.4f  iou=%.4f  px_acc=%.4f",
                val_metrics["loss"], val_metrics["dice"],
                val_metrics["iou"], val_metrics["pixel_accuracy"],
            )

            # ── Accumulate history ────────────────────────────────────
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_dice"].append(train_metrics["dice"])
            history["val_dice"].append(val_metrics["dice"])
            history["train_iou"].append(train_metrics["iou"])
            history["val_iou"].append(val_metrics["iou"])
            history["epochs_completed"] = epoch - start_epoch + 1

            # ── Checkpointing ─────────────────────────────────────────
            combined_metrics = {
                "train": train_metrics,
                "val":   val_metrics,
                "epoch": epoch,
            }
            is_best = val_metrics["loss"] < self._best_val_loss
            if is_best:
                self._best_val_loss = val_metrics["loss"]
                log.info("New best val loss: %.4f", self._best_val_loss)

            self.save_checkpoint(epoch, combined_metrics, is_best=is_best)

        # ── Save training history ─────────────────────────────────────
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        log.info("Training history saved -> %s", history_path)

        # ── Attempt to save a training progress plot ──────────────────
        try:
            self._save_training_plot(history)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save training plot: %s", exc)

        return history

    # ------------------------------------------------------------------
    def _save_training_plot(self, history: dict) -> None:
        """Save loss and dice curves to output_dir as a PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, history["epochs_completed"] + 1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(epochs, history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, history["train_dice"], label="Train Dice")
        axes[1].plot(epochs, history["val_dice"],   label="Val Dice")
        axes[1].set_title("Dice Coefficient")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = self.output_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info("Training progress plot saved -> %s", plot_path)
