"""
src/data/kaggle_download.py
===========================
Downloads and extracts the Oxford-IIIT Pet dataset from Kaggle.

Dataset
-------
  Owner   : zippyz
  ID      : zippyz/cats-and-dogs-breeds-classification-oxford-dataset
  URL     : https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset

Usage
-----
    python src/data/kaggle_download.py
    # or call from another script:
    from src.data.kaggle_download import download_dataset
    download_dataset()
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("kaggle_download")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "zippyz/cats-and-dogs-breeds-classification-oxford-dataset"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dataset(
    dataset_id: str = DATASET_ID,
    dest_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download and unzip the Kaggle dataset into *dest_dir*.

    Parameters
    ----------
    dataset_id:
        Kaggle dataset slug (``owner/name``).
    dest_dir:
        Destination directory.  Defaults to ``data/raw/`` relative to
        the project root.
    force:
        If *True*, re-download even if files already exist.

    Returns
    -------
    Path
        The resolved destination directory.

    Raises
    ------
    RuntimeError
        If the Kaggle CLI call fails.
    FileNotFoundError
        If the download produced no files in the destination directory.
    """
    cfg = load_config()
    if dest_dir is None:
        dest_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]

    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if already downloaded ──────────────────────────────────────────
    existing_files = list(dest_dir.rglob("*"))
    existing_non_empty = [f for f in existing_files if f.is_file()]
    if existing_non_empty and not force:
        log.info(
            f"Destination '{dest_dir}' already contains "
            f"{len(existing_non_empty)} file(s). "
            "Skipping download (pass force=True to re-download)."
        )
        return dest_dir

    # ── Run Kaggle CLI ───────────────────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", dataset_id,
        "-p", str(dest_dir),
        "--unzip",
    ]

    log.info(f"Downloading dataset '{dataset_id}' -> {dest_dir}")
    log.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,   # 1-hour timeout for large datasets
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Kaggle download timed out after 1 hour. "
            "Check your network connection and try again."
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not find Python executable '{sys.executable}': {exc}"
        ) from exc

    # Log stdout / stderr regardless of exit code
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log.info(f"[kaggle] {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log.warning(f"[kaggle-err] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle CLI exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}\n"
            "Possible causes:\n"
            "  1. kaggle.json credentials are missing or invalid.\n"
            "  2. You have not accepted the dataset terms on Kaggle.\n"
            "  3. Network connectivity issues."
        )

    # ── Post-download verification ───────────────────────────────────────────
    _verify_download(dest_dir)
    return dest_dir


def _verify_download(dest_dir: Path) -> None:
    """Raise FileNotFoundError if the destination is empty after download."""
    all_files = [f for f in dest_dir.rglob("*") if f.is_file()]
    if not all_files:
        raise FileNotFoundError(
            f"Download appears to have succeeded (exit code 0) but "
            f"'{dest_dir}' contains no files. "
            "This may indicate an extraction failure."
        )

    total_bytes = sum(f.stat().st_size for f in all_files)
    log.info(
        f"Download complete. "
        f"{len(all_files)} file(s), "
        f"{total_bytes / 1_048_576:.1f} MB total."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the Oxford-IIIT Pet dataset from Kaggle."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory (default: data/raw/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if destination already has files.",
    )
    args = parser.parse_args()

    try:
        out = download_dataset(dest_dir=args.dest, force=args.force)
        log.info(f"Dataset ready at: {out}")
        sys.exit(0)
    except Exception as exc:
        log.error(f"Download failed: {exc}")
        sys.exit(1)
