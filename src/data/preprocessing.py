"""
src/data/preprocessing.py
=========================
Preprocessing pipeline that reads the Oxford-IIIT Pet dataset annotation list,
matches image+trimap pairs, performs a stratified train/val/test split, and saves
CSV split files plus a JSON stats file.

Usage
-----
    from src.data.preprocessing import run_preprocessing
    stats = run_preprocessing()

    # or
    python src/data/preprocessing.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as __main__
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_preprocessing(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Execute the full preprocessing pipeline.

    Returns
    -------
    dict
        Dataset statistics dictionary saved to ``data/splits/dataset_stats.json``.
    """
    cfg = load_config(config_path)
    project_root = _PROJECT_ROOT

    # ------------------------------------------------------------------
    # Resolve directories from config
    # ------------------------------------------------------------------
    raw_dir         = project_root / cfg["dataset"]["raw_dir"]
    splits_dir      = project_root / cfg["dataset"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Actual image/mask directories (note nested layout from Kaggle download)
    images_dir = raw_dir / "images" / "images"
    masks_dir  = raw_dir / "annotations" / "annotations" / "trimaps"
    list_file  = raw_dir / "annotations" / "annotations" / "list.txt"

    log.info("Images dir  : %s", images_dir)
    log.info("Masks  dir  : %s", masks_dir)
    log.info("List   file : %s", list_file)

    if not list_file.exists():
        raise FileNotFoundError(f"Annotation list not found: {list_file}")

    # ------------------------------------------------------------------
    # Parse list.txt
    # Format: <image_id> <class_id> <species> <breed_id>
    #   species: 1=cat, 2=dog
    # ------------------------------------------------------------------
    records: list[dict] = []
    with open(list_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            image_id  = parts[0]   # e.g. "Abyssinian_1"
            # species from list: 1=cat, 2=dog
            species_id = int(parts[2])
            # Derive breed from image_id: everything before last underscore
            breed = "_".join(image_id.split("_")[:-1])
            # Cross-check with naming convention:
            # uppercase first letter = cat, lowercase = dog
            species = "cat" if image_id[0].isupper() else "dog"
            records.append({
                "image_id":  image_id,
                "species":   species,
                "breed":     breed,
            })

    log.info("Parsed %d records from list.txt", len(records))

    # ------------------------------------------------------------------
    # Match image + trimap pairs
    # ------------------------------------------------------------------
    # Build lookup dicts keyed by stem
    image_stems = {p.stem: p for p in images_dir.glob("*.jpg")}
    mask_stems  = {p.stem: p for p in masks_dir.glob("*.png")}

    matched: list[dict] = []
    for rec in records:
        stem = rec["image_id"]
        if stem in image_stems and stem in mask_stems:
            # Store relative paths from project root
            img_rel  = image_stems[stem].relative_to(project_root).as_posix()
            mask_rel = mask_stems[stem].relative_to(project_root).as_posix()
            matched.append({
                "image_path": img_rel,
                "mask_path":  mask_rel,
                "species":    rec["species"],
                "breed":      rec["breed"],
            })
        else:
            log.debug("No match for stem: %s", stem)

    log.info("Matched %d image+mask pairs", len(matched))

    if not matched:
        raise RuntimeError("No matched image+mask pairs found. Check directory layout.")

    df = pd.DataFrame(matched)

    # ------------------------------------------------------------------
    # Stratified split 70 / 15 / 15 by species
    # ------------------------------------------------------------------
    seed = cfg.get("seed", 42)
    labels = df["species"].values

    # First split: train vs (val+test)
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed
    )
    train_idx, temp_idx = next(splitter1.split(df, labels))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp  = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: val vs test  (50/50 of the 30% → 15/15 each)
    temp_labels = df_temp["species"].values
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    )
    val_idx, test_idx = next(splitter2.split(df_temp, temp_labels))

    df_val  = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    log.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(df_train), len(df_val), len(df_test),
    )

    # ------------------------------------------------------------------
    # Save CSV files
    # ------------------------------------------------------------------
    cols = ["image_path", "mask_path", "species", "breed"]
    df_train[cols].to_csv(splits_dir / "train.csv", index=False)
    df_val[cols].to_csv(splits_dir  / "val.csv",   index=False)
    df_test[cols].to_csv(splits_dir / "test.csv",  index=False)
    log.info("Saved split CSVs to %s", splits_dir)

    # ------------------------------------------------------------------
    # Save stats JSON
    # ------------------------------------------------------------------
    cat_count = int((df["species"] == "cat").sum())
    dog_count = int((df["species"] == "dog").sum())

    stats: dict[str, Any] = {
        "total_samples": len(df),
        "train_count":   len(df_train),
        "val_count":     len(df_val),
        "test_count":    len(df_test),
        "cat_count":     cat_count,
        "dog_count":     dog_count,
        "image_dir":     images_dir.as_posix(),
        "mask_dir":      masks_dir.as_posix(),
    }

    stats_path = splits_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log.info("Saved dataset stats to %s", stats_path)
    log.info("Stats: %s", stats)

    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stats = run_preprocessing()
    print("\n=== Dataset Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
