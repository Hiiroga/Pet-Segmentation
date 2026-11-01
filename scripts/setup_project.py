#!/usr/bin/env python
"""
scripts/setup_project.py
========================
One-time project initialisation script.

Responsibilities
----------------
1. Load and validate the configuration file.
2. Create all required output / checkpoint directories.
3. Set the random seed.
4. Print a summary of resolved paths and device.

Run from the project root:
    python scripts/setup_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_project_root
from src.utils.reproducibility import set_seed, get_device
from src.utils.logger import configure_root_logger, get_logger
from src.utils.directories import ensure_dirs


def main() -> None:
    cfg = load_config()
    configure_root_logger(
        level=cfg["logging"]["level"],
        log_file=PROJECT_ROOT / cfg["logging"]["log_dir"] / cfg["logging"]["log_file"],
    )
    log = get_logger("setup")

    log.info("=" * 60)
    log.info("Pet Semantic Segmentation — Project Setup")
    log.info("=" * 60)

    # ── Seed ────────────────────────────────────────────────────────────────
    seed = cfg["seed"]
    set_seed(seed)
    log.info(f"Random seed set to {seed}")

    # ── Device ──────────────────────────────────────────────────────────────
    device = get_device(cfg["device"])
    log.info(f"Compute device: {device}")

    # ── Directories ─────────────────────────────────────────────────────────
    ensure_dirs(cfg, project_root=PROJECT_ROOT)
    log.info("All output directories created / verified.")

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info(f"Project root  : {PROJECT_ROOT}")
    log.info(f"Image size    : {cfg['model']['image_size']}x{cfg['model']['image_size']}")
    log.info(f"Batch size    : {cfg['training']['batch_size']}")
    log.info(f"Learning rate : {cfg['training']['learning_rate']}")
    log.info(f"Epochs        : {cfg['training']['epochs']}")
    log.info(f"Weight decay  : {cfg['training']['weight_decay']}")
    log.info(f"Threshold     : {cfg['inference']['threshold']}")
    log.info("-" * 60)
    log.info("Setup complete. Ready to proceed with dataset download.")


if __name__ == "__main__":
    main()

