"""
src/inference/predictor.py
==========================
Model inference pipeline: load checkpoint → predict binary mask.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Union

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import get_device
from src.utils.logger import get_logger

log = get_logger("predictor")

# ImageNet normalisation constants (must match training)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class Predictor:
    """
    Load a trained U-Net checkpoint and run inference on single images.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a ``.pth`` checkpoint saved by ``Trainer``.
    device_cfg : str
        ``"auto"`` | ``"cuda"`` | ``"cpu"``
    threshold : float
        Sigmoid probability cut-off for binary mask (default 0.5).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, pathlib.Path],
        device_cfg: str = "auto",
        threshold: float = 0.5,
    ) -> None:
        from src.models.unet import UNet

        self.device = get_device(device_cfg)
        self.threshold = threshold
        self.cfg = load_config()

        checkpoint_path = pathlib.Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        log.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Support both raw state_dict and full checkpoint dicts
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        cfg_model = self.cfg.get("model", {})
        base_features = tuple(cfg_model.get("base_features", [64, 128, 256, 512]))

        self.model = UNet(
            in_channels=cfg_model.get("in_channels", 3),
            out_channels=cfg_model.get("out_channels", 1),
            base_features=base_features,
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.image_size = cfg_model.get("image_size", 256)
        log.info(f"Model loaded. Device: {self.device}, Threshold: {self.threshold}")

    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a normalised tensor [1,3,H,W]."""
        img = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0

        mean = np.array(_MEAN, dtype=np.float32)
        std  = np.array(_STD,  dtype=np.float32)
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        return tensor.to(self.device)

    def predict(
        self,
        image: Union[Image.Image, str, pathlib.Path, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single image.

        Parameters
        ----------
        image : PIL Image | path | numpy array (H,W,3 uint8)

        Returns
        -------
        binary_mask : np.ndarray, shape (H, W), dtype uint8 — 0 or 255
        probability_map : np.ndarray, shape (H, W), dtype float32 in [0,1]
        """
        if isinstance(image, (str, pathlib.Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        original_size = image.size  # (W, H)
        tensor = self._preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)                      # [1,1,H,W]
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)

        # Resize probability map back to original size
        prob_img = Image.fromarray((probs * 255).astype(np.uint8)).resize(
            original_size, Image.BILINEAR
        )
        prob_resized = np.array(prob_img).astype(np.float32) / 255.0

        binary = (prob_resized >= self.threshold).astype(np.uint8) * 255
        return binary, prob_resized

    def predict_with_overlay(
        self,
        image: Union[Image.Image, str, pathlib.Path, np.ndarray],
        alpha: float = 0.5,
        overlay_color: tuple[int, int, int] = (0, 255, 0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (binary_mask, probability_map, overlay_image).

        overlay_image : np.ndarray (H, W, 3) uint8 — image with green FG highlight.
        """
        if isinstance(image, (str, pathlib.Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        img_arr = np.array(image)
        binary, prob = self.predict(image)

        overlay = img_arr.copy()
        fg = binary == 255
        color = np.array(overlay_color, dtype=np.float32)
        overlay[fg] = (
            (1 - alpha) * img_arr[fg].astype(np.float32) + alpha * color
        ).clip(0, 255).astype(np.uint8)

        return binary, prob, overlay

