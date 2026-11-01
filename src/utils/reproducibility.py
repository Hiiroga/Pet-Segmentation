"""
utils/reproducibility.py
========================
Centralised random-seed and device selection utilities.

Usage
-----
    from src.utils.reproducibility import set_seed, get_device

    set_seed(42)
    device = get_device()          # torch.device("cuda") or ("cpu")
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fix all relevant random-number generators for reproducibility.

    Sets seeds for:
    - Python's built-in ``random`` module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variable ``PYTHONHASHSEED``

    Also enforces deterministic CUDA algorithms where possible.

    Parameters
    ----------
    seed:
        Integer seed value.  Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU setups
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Request deterministic algorithm selection.
    # Note: some ops may not have deterministic implementations →
    # we fall back gracefully by not raising errors.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # reproducibility > speed


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(device_cfg: str = "auto") -> torch.device:
    """
    Resolve a ``torch.device`` from a configuration string.

    Parameters
    ----------
    device_cfg:
        One of:

        * ``"auto"``  — select CUDA if available, else CPU *(default)*
        * ``"cuda"``  — force CUDA (raises RuntimeError if unavailable)
        * ``"cpu"``   — force CPU

    Returns
    -------
    torch.device

    Raises
    ------
    RuntimeError
        If ``device_cfg == "cuda"`` but CUDA is not available.
    ValueError
        If an unrecognised device string is passed.
    """
    device_cfg = device_cfg.strip().lower()

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but CUDA is not available on "
                "this machine.  Set device='cpu' or device='auto' in "
                "configs/config.yaml."
            )
        return torch.device("cuda")

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device configuration '{device_cfg}'.  "
        "Expected 'auto', 'cuda', or 'cpu'."
    )

