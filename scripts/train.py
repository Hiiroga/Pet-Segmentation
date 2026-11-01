"""
scripts/train.py
================
Training entry-point for the Pet Semantic Segmentation project.

Usage
-----
    python scripts/train.py           # fresh training
    python scripts/train.py --resume  # resume from latest checkpoint
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from src.data.dataset import get_dataloaders
from src.losses.combined_loss import CombinedLoss
from src.models.unet import UNet
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(cfg: dict) -> torch.device:
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    return device


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recently saved epoch checkpoint, or None."""
    pattern = str(checkpoint_dir / "checkpoint_epoch_*.pth")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    return Path(files[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (optional)"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    cfg = load_config(args.config)
    log.info("Config loaded.")

    # ── Reproducibility ───────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    set_seed(seed)
    log.info("Seed set to %d", seed)

    # ── Device ────────────────────────────────────────────────────────
    device = _get_device(cfg)
    log.info("Using device: %s", device)

    # cuDNN benchmark — finds fastest conv algorithm for fixed input size
    if device.type == "cuda":
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        log.info("cuDNN benchmark mode enabled.")

    # ── DataLoaders ───────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = UNet(
        in_channels   = model_cfg.get("in_channels", 3),
        out_channels  = model_cfg.get("out_channels", 1),
        base_features = model_cfg.get("base_features", [64, 128, 256, 512]),
    ).to(device)
    log.info("Model created — trainable params: %d", model.count_parameters())

    # ── Loss ──────────────────────────────────────────────────────────
    loss_cfg = cfg["loss"]
    criterion = CombinedLoss(
        bce_weight  = loss_cfg.get("bce_weight",  0.5),
        dice_weight = loss_cfg.get("dice_weight", 0.5),
        smooth      = loss_cfg.get("dice_smooth", 1.0),
    )

    # ── Optimizer ─────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    opt_cfg   = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = train_cfg.get("learning_rate", 0.001),
        weight_decay = train_cfg.get("weight_decay",  0.0001),
        betas        = tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps          = opt_cfg.get("eps", 1e-8),
    )

    # ── Paths ─────────────────────────────────────────────────────────
    checkpoint_dir = _PROJECT_ROOT / cfg["checkpoints"]["dir"]
    output_dir     = _PROJECT_ROOT / cfg["outputs"].get("training_progress",
                                                         "outputs/training_progress")

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        optimizer      = optimizer,
        criterion      = criterion,
        device         = device,
        cfg            = cfg,
        checkpoint_dir = checkpoint_dir,
        output_dir     = output_dir,
    )

    # ── Optional resume ───────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        latest_ckpt = _find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            ckpt = trainer.load_checkpoint(latest_ckpt)
            start_epoch = ckpt["epoch"] + 1
            log.info("Resuming from epoch %d", start_epoch)
        else:
            log.warning("--resume specified but no checkpoint found; starting fresh.")

    # ── Train ─────────────────────────────────────────────────────────
    history = trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    log.info("Training complete. Epochs completed: %d", history["epochs_completed"])
    log.info(
        "Final train loss=%.4f  val loss=%.4f",
        history["train_loss"][-1] if history["train_loss"] else float("nan"),
        history["val_loss"][-1]   if history["val_loss"]   else float("nan"),
    )


if __name__ == "__main__":
    main()
