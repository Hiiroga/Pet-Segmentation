#!/usr/bin/env python
"""
scripts/preprocess.py
=====================
Entry-point for the preprocessing pipeline.

Run from project root:
    python scripts/preprocess.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import run_preprocessing
from src.utils.logger import get_logger

log = get_logger("preprocess_script")

if __name__ == "__main__":
    log.info("Starting preprocessing pipeline ...")
    stats = run_preprocessing()
    log.info("Preprocessing complete.")
    log.info(f"  Total samples : {stats['total_samples']:,}")
    log.info(f"  Train         : {stats['train_count']:,}")
    log.info(f"  Val           : {stats['val_count']:,}")
    log.info(f"  Test          : {stats['test_count']:,}")
    log.info(f"  Cats          : {stats['cat_count']:,}")
    log.info(f"  Dogs          : {stats['dog_count']:,}")

