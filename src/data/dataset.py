"""
src/data/dataset.py
===================
PyTorch Dataset and DataLoader factories for the Oxford-IIIT Pet segmentation task.

Trimap label convention (raw PNG values):
  1 → Foreground → mask value 1.0
  2 → Background → mask value 0.0
  3 → Boundary   → treated as background → mask value 0.0

Augmentation uses only torchvision transforms — no albumentations.
Geometric transforms (flip, rotation) are applied identically to both
the image and the mask to keep alignment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when used standalone
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ImageNet normalisation statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PetSegmentationDataset(Dataset):
    """
    Oxford-IIIT Pet segmentation dataset.

    Parameters
    ----------
    csv_path : str | Path
        Path to a split CSV file (train.csv / val.csv / test.csv) with columns:
        image_path, mask_path, species, breed.
    image_size : int
        Square spatial resolution to resize images and masks to.
    augment : bool
        Whether to apply random augmentation (use True only for training).
    seed : int
        Random seed for reproducible augmentation decisions.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.csv_path   = Path(csv_path)
        self.image_size = image_size
        self.augment    = augment
        self.seed       = seed

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV split file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        log.info(
            "Loaded %d samples from %s (augment=%s)",
            len(self.df), self.csv_path.name, augment,
        )

        # Normalisation transform for images only
        self._normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        image : torch.Tensor  float32 [3, H, W]  — ImageNet-normalised
        mask  : torch.Tensor  float32 [1, H, W]  — binary {0.0, 1.0}
        """
        row = self.df.iloc[idx]

        # Resolve paths relative to project root
        img_path  = _PROJECT_ROOT / row["image_path"]
        mask_path = _PROJECT_ROOT / row["mask_path"]

        # Load image as RGB PIL
        image = Image.open(img_path).convert("RGB")

        # Load trimap as grayscale (values: 1, 2, 3)
        mask = Image.open(mask_path).convert("L")

        # Resize to target size (image: bilinear, mask: nearest to preserve labels)
        image = TF.resize(image, [self.image_size, self.image_size],
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  [self.image_size, self.image_size],
                          interpolation=TF.InterpolationMode.NEAREST)

        # ── Augmentation (training only) ──────────────────────────────
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # ── Convert image to tensor and normalise ─────────────────────
        image_tensor = TF.to_tensor(image)          # [3, H, W], float32 in [0,1]
        image_tensor = self._normalize(image_tensor)

        # ── Convert mask: trimap → binary float ───────────────────────
        # PIL "L" → torch uint8 then cast; shape [H, W]
        mask_np = torch.from_numpy(
            __import__("numpy").array(mask, dtype="int32")
        )
        # 1=FG→1.0, 2=BG→0.0, 3=Boundary→0.0
        binary_mask = (mask_np == 1).float()        # [H, W]
        mask_tensor = binary_mask.unsqueeze(0)      # [1, H, W]

        return image_tensor, mask_tensor

    # ------------------------------------------------------------------
    def _apply_augmentation(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply random geometric transforms identically to image and mask,
        and colour jitter to the image only."""

        # Random horizontal flip (p=0.5)
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip (p=0.3)
        if random.random() < 0.3:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation ±15 degrees (p=0.5)
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR,
                              fill=0)
            mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST,
                              fill=0)

        # Colour jitter (image only)
        jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        )
        image = jitter(image)

        return image, mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader) from config.

    Parameters
    ----------
    cfg : dict
        Parsed config dictionary (from ``load_config()``).

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
    """
    splits_dir = _PROJECT_ROOT / cfg["dataset"]["splits_dir"]
    image_size = cfg["model"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)
    pin_memory  = cfg["training"].get("pin_memory", False)
    seed        = cfg.get("seed", 42)

    # On Windows, num_workers > 0 can cause issues; cap at 0 as safe fallback
    if os.name == "nt":
        num_workers = 0

    train_ds = PetSegmentationDataset(
        csv_path   = splits_dir / "train.csv",
        image_size = image_size,
        augment    = True,
        seed       = seed,
    )
    val_ds = PetSegmentationDataset(
        csv_path   = splits_dir / "val.csv",
        image_size = image_size,
        augment    = False,
        seed       = seed,
    )
    test_ds = PetSegmentationDataset(
        csv_path   = splits_dir / "test.csv",
        image_size = image_size,
        augment    = False,
        seed       = seed,
    )

    # Reproducible worker seeding
    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        import numpy as np
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        worker_init_fn = _seed_worker,
        generator   = g,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    log.info(
        "DataLoaders ready — train: %d batches | val: %d batches | test: %d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    imgs, masks = next(iter(train_loader))
    print(f"Image batch : {imgs.shape}  dtype={imgs.dtype}")
    print(f"Mask  batch : {masks.shape} dtype={masks.dtype}")
    print(f"Mask  unique values: {masks.unique()}")
