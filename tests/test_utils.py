"""
tests/test_utils.py
===================
Unit tests for the shared utility modules:
  - src.utils.config        (load_config, get_project_root)
  - src.utils.reproducibility (set_seed, get_device)
  - src.utils.logger        (get_logger)
  - src.utils.directories   (ensure_dirs, make_dir)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ===========================================================================
# Config tests
# ===========================================================================

class TestLoadConfig:
    def test_loads_default_config(self):
        """load_config() without arguments should succeed."""
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert isinstance(cfg, dict)

    def test_required_sections_present(self):
        """All required top-level sections must be present."""
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        required = [
            "dataset", "model", "training", "loss",
            "optimizer", "seed", "device", "checkpoints",
            "outputs", "inference",
        ]
        for section in required:
            assert section in cfg, f"Missing section: {section}"

    def test_image_size(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert cfg["model"]["image_size"] == 256

    def test_batch_size(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert cfg["training"]["batch_size"] == 8

    def test_learning_rate(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert abs(cfg["training"]["learning_rate"] - 0.001) < 1e-9

    def test_epochs(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert cfg["training"]["epochs"] == 20

    def test_seed(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert cfg["seed"] == 42

    def test_weight_decay(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert abs(cfg["training"]["weight_decay"] - 0.0001) < 1e-9

    def test_early_stopping_disabled(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert cfg["training"]["early_stopping"] is False

    def test_inference_threshold(self):
        from src.utils.config import load_config
        cfg = load_config(CONFIG_PATH)
        assert abs(cfg["inference"]["threshold"] - 0.5) < 1e-9

    def test_raises_on_missing_file(self):
        from src.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_get_project_root_is_directory(self):
        from src.utils.config import get_project_root
        root = get_project_root()
        assert root.is_dir()


# ===========================================================================
# Reproducibility tests
# ===========================================================================

class TestSetSeed:
    def test_set_seed_runs_without_error(self):
        from src.utils.reproducibility import set_seed
        set_seed(42)

    def test_torch_reproducibility(self):
        """Same seed → same random tensor."""
        from src.utils.reproducibility import set_seed
        set_seed(0)
        t1 = torch.rand(5)
        set_seed(0)
        t2 = torch.rand(5)
        assert torch.allclose(t1, t2)

    def test_pythonhashseed_env_var(self):
        from src.utils.reproducibility import set_seed
        set_seed(99)
        assert os.environ.get("PYTHONHASHSEED") == "99"


class TestGetDevice:
    def test_auto_returns_device(self):
        from src.utils.reproducibility import get_device
        device = get_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "cpu")

    def test_cpu_returns_cpu(self):
        from src.utils.reproducibility import get_device
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_invalid_device_raises(self):
        from src.utils.reproducibility import get_device
        with pytest.raises(ValueError):
            get_device("tpu")

    def test_cuda_raises_when_unavailable(self):
        from src.utils.reproducibility import get_device
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError):
                get_device("cuda")


# ===========================================================================
# Logger tests
# ===========================================================================

class TestGetLogger:
    def test_returns_logger_instance(self):
        from src.utils.logger import get_logger
        log = get_logger("test_logger_A")
        assert isinstance(log, logging.Logger)

    def test_logger_level_info(self):
        from src.utils.logger import get_logger
        log = get_logger("test_logger_B", level="INFO")
        assert log.level == logging.INFO

    def test_logger_level_debug(self):
        from src.utils.logger import get_logger
        log = get_logger("test_logger_C", level="DEBUG")
        assert log.level == logging.DEBUG

    def test_file_handler_created(self):
        """Logger creates the log file; handler is explicitly closed on Windows."""
        import logging
        from src.utils.logger import get_logger
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = get_logger("test_logger_D", log_file=log_path)
            assert log_path.exists()
            # Close file handlers so Windows releases the file lock before cleanup.
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)


# ===========================================================================
# Directory tests
# ===========================================================================

class TestDirectories:
    def test_make_dir_creates_directory(self):
        from src.utils.directories import make_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "a" / "b" / "c"
            result = make_dir(new_dir)
            assert result.is_dir()

    def test_make_dir_returns_path(self):
        from src.utils.directories import make_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = make_dir(Path(tmpdir) / "nested")
            assert isinstance(p, Path)

    def test_ensure_dirs_creates_output_dirs(self):
        from src.utils.config import load_config
        from src.utils.directories import ensure_dirs
        cfg = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_dirs(cfg, project_root=Path(tmpdir))
            assert (Path(tmpdir) / cfg["checkpoints"]["dir"]).is_dir()
            assert (Path(tmpdir) / cfg["outputs"]["dir"]).is_dir()
