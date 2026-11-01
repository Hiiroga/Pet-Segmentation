"""
tests/test_inference.py
=======================
Stage 09 – Automated tests for the Predictor class and end-to-end inference.

These tests are designed to run AFTER training is complete (best_model.pth exists).
If the checkpoint is missing the tests are skipped gracefully with pytest.skip().
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "best_model.pth"


def _require_checkpoint():
    """Skip test if best_model.pth does not exist yet."""
    if not CHECKPOINT.exists():
        pytest.skip("best_model.pth not found – run training first.")


def _make_dummy_image(size: int = 256) -> Image.Image:
    """Create a random RGB PIL image."""
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Predictor instantiation tests
# ---------------------------------------------------------------------------

class TestPredictorLoad:
    def test_loads_checkpoint(self):
        _require_checkpoint()
        from src.inference.predictor import Predictor
        predictor = Predictor(CHECKPOINT, device_cfg="cpu")
        assert predictor.model is not None

    def test_model_in_eval_mode(self):
        _require_checkpoint()
        from src.inference.predictor import Predictor
        predictor = Predictor(CHECKPOINT, device_cfg="cpu")
        assert not predictor.model.training, "Model must be in eval() mode after loading."

    def test_missing_checkpoint_raises(self):
        from src.inference.predictor import Predictor
        with pytest.raises(FileNotFoundError):
            Predictor("nonexistent/checkpoint.pth", device_cfg="cpu")

    def test_threshold_stored(self):
        _require_checkpoint()
        from src.inference.predictor import Predictor
        predictor = Predictor(CHECKPOINT, device_cfg="cpu", threshold=0.7)
        assert predictor.threshold == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Prediction output tests
# ---------------------------------------------------------------------------

class TestPredictorOutput:
    @pytest.fixture(scope="class")
    def predictor(self):
        _require_checkpoint()
        from src.inference.predictor import Predictor
        return Predictor(CHECKPOINT, device_cfg="cpu")

    def test_predict_returns_two_arrays(self, predictor):
        img = _make_dummy_image()
        result = predictor.predict(img)
        assert len(result) == 2, "predict() must return (binary_mask, probability_map)"

    def test_binary_mask_dtype(self, predictor):
        img = _make_dummy_image()
        binary, _ = predictor.predict(img)
        assert binary.dtype == np.uint8

    def test_binary_mask_values(self, predictor):
        img = _make_dummy_image()
        binary, _ = predictor.predict(img)
        unique_vals = set(np.unique(binary).tolist())
        assert unique_vals.issubset({0, 255}), f"Unexpected binary values: {unique_vals}"

    def test_probability_map_range(self, predictor):
        img = _make_dummy_image()
        _, prob = predictor.predict(img)
        assert prob.min() >= 0.0 - 1e-6
        assert prob.max() <= 1.0 + 1e-6

    def test_probability_map_dtype(self, predictor):
        img = _make_dummy_image()
        _, prob = predictor.predict(img)
        assert prob.dtype == np.float32

    def test_output_shape_matches_input(self, predictor):
        """The output should be resized back to the input image dimensions."""
        img = _make_dummy_image(size=300)
        binary, prob = predictor.predict(img)
        assert binary.shape == (300, 300)
        assert prob.shape   == (300, 300)

    def test_predict_from_path(self, tmp_path, predictor):
        """Predictor should accept a file path string."""
        img = _make_dummy_image()
        img_path = tmp_path / "test_img.jpg"
        img.save(img_path)
        binary, prob = predictor.predict(str(img_path))
        assert binary.ndim == 2
        assert prob.ndim   == 2

    def test_predict_from_numpy(self, predictor):
        """Predictor should accept a numpy uint8 array."""
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        binary, prob = predictor.predict(arr)
        assert binary.shape == (256, 256)

    def test_threshold_affects_binary(self, predictor):
        """High threshold -> fewer foreground pixels; low threshold -> more."""
        from src.inference.predictor import Predictor
        img = _make_dummy_image()

        pred_high = Predictor(CHECKPOINT, device_cfg="cpu", threshold=0.9)
        pred_low  = Predictor(CHECKPOINT, device_cfg="cpu", threshold=0.1)

        binary_high, _ = pred_high.predict(img)
        binary_low,  _ = pred_low.predict(img)

        fg_high = (binary_high == 255).sum()
        fg_low  = (binary_low  == 255).sum()
        assert fg_low >= fg_high, "Lower threshold must produce >= foreground pixels."


# ---------------------------------------------------------------------------
# predict_with_overlay tests
# ---------------------------------------------------------------------------

class TestPredictWithOverlay:
    @pytest.fixture(scope="class")
    def predictor(self):
        _require_checkpoint()
        from src.inference.predictor import Predictor
        return Predictor(CHECKPOINT, device_cfg="cpu")

    def test_returns_three_items(self, predictor):
        img = _make_dummy_image()
        result = predictor.predict_with_overlay(img)
        assert len(result) == 3

    def test_overlay_shape(self, predictor):
        img = _make_dummy_image(size=256)
        binary, prob, overlay = predictor.predict_with_overlay(img)
        assert overlay.shape == (256, 256, 3)

    def test_overlay_dtype(self, predictor):
        img = _make_dummy_image()
        _, _, overlay = predictor.predict_with_overlay(img)
        assert overlay.dtype == np.uint8

    def test_overlay_value_range(self, predictor):
        img = _make_dummy_image()
        _, _, overlay = predictor.predict_with_overlay(img)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

