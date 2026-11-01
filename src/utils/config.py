"""
utils/config.py
===============
Loads and validates the YAML configuration file, exposing a typed
Config object consumed across the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the YAML configuration file and return it as a plain dict.

    Parameters
    ----------
    config_path:
        Path to the YAML file.  If *None*, the function looks for
        ``configs/config.yaml`` relative to the project root
        (detected by walking up from this file's location).

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    if config_path is None:
        # Walk up from src/utils/ → src/ → project root
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure you run the project from the project root or pass an "
            "explicit config_path."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    _validate_config(cfg)
    return cfg


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_config(cfg: dict[str, Any]) -> None:
    """Raise *ValueError* when required top-level sections are missing."""
    required_sections = [
        "dataset", "model", "training", "loss",
        "optimizer", "seed", "device", "checkpoints",
        "outputs", "inference",
    ]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(
            f"Configuration is missing required sections: {missing}"
        )

