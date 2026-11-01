"""
src/data/inspect_dataset.py
===========================
Recursively inspects the extracted Oxford-IIIT Pet dataset, verifies
image-mask pairs, validates trimap pixel values, and generates a
dataset report + sample visualizations.

Usage
-----
    python src/data/inspect_dataset.py
"""

from __future__ import annotations

import collections
import pathlib
import sys
from typing import Any

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("inspect_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_TRIMAP_VALUES = {1, 2, 3}   # 1=FG, 2=BG, 3=Boundary
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS    = {".png"}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_images(image_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return all non-macOS-metadata image files under *image_dir*."""
    files = [
        f for f in image_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.name.startswith("._")
    ]
    return sorted(files)


def find_masks(mask_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return all non-macOS-metadata mask (trimap) files under *mask_dir*."""
    files = [
        f for f in mask_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in MASK_EXTENSIONS
        and not f.name.startswith("._")
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------

def build_pairs(
    image_files: list[pathlib.Path],
    mask_files: list[pathlib.Path],
) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], list[pathlib.Path], list[pathlib.Path]]:
    """
    Match images to masks by stem name.

    Returns
    -------
    paired : list of (image_path, mask_path)
    images_without_mask : list of image paths that have no matching mask
    masks_without_image : list of mask paths that have no matching image
    """
    mask_index = {m.stem: m for m in mask_files}
    image_index = {i.stem: i for i in image_files}

    paired = []
    images_without_mask = []
    masks_without_image = []

    for img in image_files:
        if img.stem in mask_index:
            paired.append((img, mask_index[img.stem]))
        else:
            images_without_mask.append(img)

    for mask in mask_files:
        if mask.stem not in image_index:
            masks_without_image.append(mask)

    return paired, images_without_mask, masks_without_image


# ---------------------------------------------------------------------------
# Trimap validation
# ---------------------------------------------------------------------------

def validate_trimap(mask_path: pathlib.Path) -> dict[str, Any]:
    """
    Open a trimap PNG and validate its pixel values.

    Returns a dict with keys:
        valid, mode, size, unique_values, has_fg, has_bg, error
    """
    result: dict[str, Any] = {
        "valid": False, "mode": None, "size": None,
        "unique_values": set(), "has_fg": False,
        "has_bg": False, "error": None,
    }
    try:
        img = Image.open(mask_path)
        arr = np.array(img)
        unique = set(int(v) for v in np.unique(arr))
        result["mode"] = img.mode
        result["size"] = img.size
        result["unique_values"] = unique
        result["has_fg"] = 1 in unique
        result["has_bg"] = 2 in unique
        # Valid = all pixel values are in {1,2,3} AND contains FG and BG
        result["valid"] = unique.issubset(VALID_TRIMAP_VALUES) and result["has_fg"] and result["has_bg"]
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def validate_image(img_path: pathlib.Path) -> dict[str, Any]:
    """Open an image and check it is readable and RGB-convertible."""
    result: dict[str, Any] = {
        "valid": False, "mode": None, "size": None, "error": None,
    }
    try:
        img = Image.open(img_path)
        img.verify()         # detect truncated / corrupted headers
        img = Image.open(img_path).convert("RGB")   # re-open after verify
        result["mode"] = img.mode
        result["size"] = img.size
        result["valid"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_sample_visualizations(
    pairs: list[tuple[pathlib.Path, pathlib.Path]],
    out_dir: pathlib.Path,
    n: int = 5,
    seed: int = 42,
) -> list[pathlib.Path]:
    """
    Save *n* sample visualizations: original | mask | overlay.

    Returns list of saved file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    saved = []

    COLORMAP = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}  # FG/BG/Boundary

    for i, idx in enumerate(indices):
        img_path, mask_path = pairs[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            mask_arr = np.array(Image.open(mask_path))

            # Build coloured mask (RGB)
            mask_rgb = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
            for val, colour in COLORMAP.items():
                mask_rgb[mask_arr == val] = colour

            # Binary mask: foreground=white, background=black, boundary=gray
            binary = np.zeros(mask_arr.shape, dtype=np.uint8)
            binary[mask_arr == 1] = 255
            binary[mask_arr == 3] = 128

            # Overlay: blend image with foreground highlight
            img_arr = np.array(img.resize(mask_arr.shape[::-1], Image.BILINEAR))
            overlay = img_arr.copy()
            fg_pixels = mask_arr == 1
            overlay[fg_pixels] = (0.5 * img_arr[fg_pixels] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(img_arr); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(mask_rgb); axes[1].set_title("Trimap (1=FG 2=BG 3=Boundary)"); axes[1].axis("off")
            axes[2].imshow(binary, cmap="gray"); axes[2].set_title("Binary FG Mask"); axes[2].axis("off")
            axes[3].imshow(overlay); axes[3].set_title("FG Overlay"); axes[3].axis("off")

            fig.suptitle(img_path.stem, fontsize=12)
            plt.tight_layout()
            out_path = out_dir / f"sample_{i+1:02d}_{img_path.stem}.png"
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)
            log.info(f"Saved sample visualization: {out_path.name}")
        except Exception as exc:
            log.warning(f"Failed to visualize {img_path.name}: {exc}")

    return saved


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------

def inspect_dataset(
    image_dir: pathlib.Path | None = None,
    mask_dir:  pathlib.Path | None = None,
    out_dir:   pathlib.Path | None = None,
    n_validate: int = 200,
    n_visualize: int = 8,
) -> dict[str, Any]:
    """
    Full dataset inspection pipeline.

    Returns a summary dict with all counts and validation results.
    """
    cfg = load_config()

    if image_dir is None:
        image_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"] / "images" / "images"
    if mask_dir is None:
        mask_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"] / "annotations" / "annotations" / "trimaps"
    if out_dir is None:
        out_dir = PROJECT_ROOT / cfg["outputs"]["dataset_samples"]

    log.info("=" * 60)
    log.info("Dataset Inspection")
    log.info("=" * 60)
    log.info(f"Image dir : {image_dir}")
    log.info(f"Mask dir  : {mask_dir}")

    # ── Discover files ───────────────────────────────────────────────────────
    images = find_images(image_dir)
    masks  = find_masks(mask_dir)
    log.info(f"Images found : {len(images):,}")
    log.info(f"Masks found  : {len(masks):,}")

    # ── Pair matching ────────────────────────────────────────────────────────
    paired, imgs_no_mask, masks_no_img = build_pairs(images, masks)
    log.info(f"Paired       : {len(paired):,}")
    log.info(f"Images without mask : {len(imgs_no_mask)}")
    log.info(f"Masks without image : {len(masks_no_img)}")

    # ── Validate a sample of trimaps ─────────────────────────────────────────
    log.info(f"Validating {min(n_validate, len(paired))} random pairs ...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(paired), size=min(n_validate, len(paired)), replace=False)

    valid_masks = 0
    invalid_masks = 0
    corrupt_images = 0
    unique_all: set[int] = set()
    trimap_value_counter: collections.Counter = collections.Counter()

    for idx in sample_idx:
        img_path, mask_path = paired[idx]

        img_info = validate_image(img_path)
        if not img_info["valid"]:
            corrupt_images += 1
            log.warning(f"Corrupt image: {img_path.name}: {img_info['error']}")

        mask_info = validate_trimap(mask_path)
        unique_all.update(mask_info["unique_values"])
        trimap_value_counter.update(mask_info["unique_values"])
        if mask_info["valid"]:
            valid_masks += 1
        else:
            invalid_masks += 1
            if mask_info["error"]:
                log.warning(f"Corrupt mask: {mask_path.name}: {mask_info['error']}")
            elif not mask_info["unique_values"].issubset(VALID_TRIMAP_VALUES):
                log.warning(f"Unexpected pixel values in {mask_path.name}: {mask_info['unique_values']}")

    log.info(f"Valid trimaps   : {valid_masks}/{min(n_validate, len(paired))}")
    log.info(f"Invalid trimaps : {invalid_masks}/{min(n_validate, len(paired))}")
    log.info(f"Corrupt images  : {corrupt_images}/{min(n_validate, len(paired))}")
    log.info(f"All unique trimap values seen: {sorted(unique_all)}")

    # ── Visualizations ────────────────────────────────────────────────────────
    saved = save_sample_visualizations(paired, out_dir, n=n_visualize)

    summary = {
        "total_images": len(images),
        "total_masks": len(masks),
        "total_paired": len(paired),
        "images_without_mask": len(imgs_no_mask),
        "masks_without_image": len(masks_no_img),
        "valid_trimaps_sampled": valid_masks,
        "invalid_trimaps_sampled": invalid_masks,
        "corrupt_images_sampled": corrupt_images,
        "unique_trimap_values": sorted(unique_all),
        "trimap_value_counts": dict(trimap_value_counter),
        "visualizations_saved": [str(p) for p in saved],
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
    }

    return summary


if __name__ == "__main__":
    result = inspect_dataset()
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    for k, v in result.items():
        if k not in ("visualizations_saved",):
            log.info(f"  {k}: {v}")
    log.info(f"  visualizations_saved: {len(result['visualizations_saved'])} files")

