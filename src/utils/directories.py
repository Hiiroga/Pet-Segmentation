"""
utils/directories.py
====================
Helpers to create and validate project output directories at runtime.

Usage
-----
    from src.utils.directories import ensure_dirs

    ensure_dirs(cfg)   # cfg = load_config(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_dirs(cfg: dict[str, Any], project_root: Path | None = None) -> None:
    """
    Create all output and checkpoint directories defined in the config.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary (from ``load_config``).
    project_root:
        Absolute path to the project root.  If *None*, uses the current
        working directory.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()

    paths_to_create: list[str] = [
        cfg["checkpoints"]["dir"],
        cfg["outputs"]["dir"],
        cfg["outputs"]["dataset_samples"],
        cfg["outputs"]["training_progress"],
        cfg["outputs"]["predictions"],
        cfg["outputs"]["overlays"],
        cfg["outputs"]["evaluation"],
        cfg["dataset"]["raw_dir"],
        cfg["dataset"]["processed_dir"],
        cfg["dataset"]["splits_dir"],
    ]

    for rel_path in paths_to_create:
        abs_path = root / rel_path
        abs_path.mkdir(parents=True, exist_ok=True)


def make_dir(path: str | Path) -> Path:
    """
    Create a single directory (and any missing parents) and return its
    resolved ``Path``.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        Resolved absolute path of the created directory.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

