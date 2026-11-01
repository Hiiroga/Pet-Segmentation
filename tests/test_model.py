"""
tests/test_model.py
===================
Unit tests for Stages 03–05:
  - src.data.preprocessing  (splits exist, correct counts)
  - src.data.dataset        (Dataset shapes, DataLoader batches)
  - src.models.unet         (UNet forward pass, parameter count)
  - src.losses.combined_loss (DiceLoss, CombinedLoss)
  - src.metrics.segmentation_metrics (SegmentationMetrics API)
"""
from __future__ import annotations

import csv
import json
import pathlib
import tempfile

import numpy as np
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
TRAIN_CSV    = SPLITS_DIR / "train.csv"
VAL_CSV      = SPLITS_DIR / "val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"
STATS_JSON   = SPLITS_DIR / "dataset_stats.json"


# ===========================================================================
# Helpers
# ===========================================================================

def _make_mock_csv(tmp_dir: pathlib.Path, n: int = 4) -> pathlib.Path:
    """Create a tiny valid CSV with real image/mask paths from the dataset."""
    csv_path = tmp_dir / "mock.csv"
    # Grab real pairs from train.csv so paths actually exist on disk
    rows = []
    with open(TRAIN_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path","mask_path","species","breed"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ===========================================================================
# Stage 03 — Preprocessing / Splits
# ===========================================================================

class TestPreprocessing:
    def test_train_csv_exists(self):
        assert TRAIN_CSV.exists(), "train.csv not found — run scripts/preprocess.py"

    def test_val_csv_exists(self):
        assert VAL_CSV.exists()

    def test_test_csv_exists(self):
        assert TEST_CSV.exists()

    def test_stats_json_exists(self):
        assert STATS_JSON.exists()

    def test_stats_keys(self):
        with open(STATS_JSON) as fh:
            stats = json.load(fh)
        for key in ("total_samples","train_count","val_count","test_count","cat_count","dog_count"):
            assert key in stats, f"Missing key: {key}"

    def test_split_counts_add_up(self):
        with open(STATS_JSON) as fh:
            stats = json.load(fh)
        total = stats["total_samples"]
        s = stats["train_count"] + stats["val_count"] + stats["test_count"]
        assert abs(s - total) <= 1, f"Split counts {s} don't match total {total}"

    def test_train_csv_has_header_and_rows(self):
        with open(TRAIN_CSV, newline="") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames) >= {"image_path","mask_path","species","breed"}
            rows = list(reader)
        assert len(rows) > 100

    def test_train_ratio_approx_70pct(self):
        with open(STATS_JSON) as fh:
            stats = json.load(fh)
        ratio = stats["train_count"] / stats["total_samples"]
        assert 0.65 <= ratio <= 0.75, f"Train ratio {ratio:.2f} outside expected range"

    def test_species_columns_valid(self):
        with open(TRAIN_CSV, newline="") as fh:
            reader = csv.DictReader(fh)
            species = {row["species"] for row in reader}
        assert species <= {"cat","dog"}, f"Unexpected species values: {species}"


# ===========================================================================
# Stage 04 — Dataset & DataLoader
# ===========================================================================

class TestDataset:
    def test_dataset_len(self):
        from src.data.dataset import PetSegmentationDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_mock_csv(pathlib.Path(tmpdir), n=4)
            ds = PetSegmentationDataset(csv_path, image_size=256, augment=False)
        assert len(ds) == 4

    def test_dataset_getitem_shapes(self):
        from src.data.dataset import PetSegmentationDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_mock_csv(pathlib.Path(tmpdir), n=4)
            ds = PetSegmentationDataset(csv_path, image_size=256, augment=False)
        img, mask = ds[0]
        assert img.shape  == (3, 256, 256), f"Image shape wrong: {img.shape}"
        assert mask.shape == (1, 256, 256), f"Mask shape wrong: {mask.shape}"

    def test_dataset_image_dtype(self):
        from src.data.dataset import PetSegmentationDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_mock_csv(pathlib.Path(tmpdir), n=2)
            ds = PetSegmentationDataset(csv_path, image_size=256, augment=False)
        img, _ = ds[0]
        assert img.dtype == torch.float32

    def test_dataset_mask_binary(self):
        from src.data.dataset import PetSegmentationDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_mock_csv(pathlib.Path(tmpdir), n=4)
            ds = PetSegmentationDataset(csv_path, image_size=256, augment=False)
        _, mask = ds[0]
        unique = torch.unique(mask).tolist()
        assert all(v in (0.0, 1.0) for v in unique), f"Mask has non-binary values: {unique}"

    def test_dataloader_batch_shape(self):
        from src.data.dataset import PetSegmentationDataset
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_mock_csv(pathlib.Path(tmpdir), n=4)
            ds = PetSegmentationDataset(csv_path, image_size=256, augment=False)
        loader = DataLoader(ds, batch_size=2)
        imgs, masks = next(iter(loader))
        assert imgs.shape  == (2, 3, 256, 256)
        assert masks.shape == (2, 1, 256, 256)

    def test_dataset_augmented_no_error(self):
        from src.data.dataset import PetSegmentationDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_mock_csv(pathlib.Path(tmpdir), n=4)
            ds = PetSegmentationDataset(csv_path, image_size=256, augment=True)
        img, mask = ds[0]
        assert img.shape  == (3, 256, 256)
        assert mask.shape == (1, 256, 256)


# ===========================================================================
# Stage 05 — U-Net Architecture
# ===========================================================================

class TestUNet:
    def test_forward_pass_shape(self):
        from src.models.unet import UNet
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert y.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {y.shape}"

    def test_output_is_logits_not_sigmoid(self):
        """Output should contain values outside [0,1] → raw logits."""
        from src.models.unet import UNet
        torch.manual_seed(0)
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        has_outside = ((y < 0) | (y > 1)).any().item()
        assert has_outside, "Output looks like probabilities, not logits"

    def test_parameter_count(self):
        from src.models.unet import UNet
        model = UNet()
        params = model.count_parameters()
        assert params > 1_000_000, f"Too few parameters: {params:,}"

    def test_batch_size_1(self):
        from src.models.unet import UNet
        model = UNet()
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        assert y.shape == (1, 1, 256, 256)

    def test_batch_size_4(self):
        from src.models.unet import UNet
        model = UNet()
        x = torch.randn(4, 3, 256, 256)
        y = model(x)
        assert y.shape == (4, 1, 256, 256)

    def test_no_nan_in_output(self):
        from src.models.unet import UNet
        torch.manual_seed(42)
        model = UNet()
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert not torch.isnan(y).any(), "NaN values in model output"

    def test_gradient_flows(self):
        from src.models.unet import UNet
        model = UNet()
        x = torch.randn(1, 3, 64, 64)   # smaller for speed
        y = model(x)
        loss = y.mean()
        loss.backward()
        # Check at least one gradient is non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradient flowed to model parameters"


# ===========================================================================
# Stage 05 — Loss Functions
# ===========================================================================

class TestDiceLoss:
    def test_output_is_scalar(self):
        from src.losses.combined_loss import DiceLoss
        loss_fn = DiceLoss()
        logits  = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, "DiceLoss should return a scalar"

    def test_perfect_prediction_low_loss(self):
        from src.losses.combined_loss import DiceLoss
        loss_fn = DiceLoss()
        # Perfect: logits very positive where target=1, very negative where target=0
        targets = torch.ones(1, 1, 32, 32)
        logits  = torch.full((1, 1, 32, 32), 10.0)   # sigmoid -> ~1.0
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05, f"Perfect prediction Dice loss too high: {loss.item():.4f}"

    def test_worst_prediction_high_loss(self):
        from src.losses.combined_loss import DiceLoss
        loss_fn = DiceLoss()
        targets = torch.ones(1, 1, 32, 32)
        logits  = torch.full((1, 1, 32, 32), -10.0)  # sigmoid -> ~0.0
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.9, f"Worst prediction Dice loss too low: {loss.item():.4f}"


class TestCombinedLoss:
    def test_returns_dict_with_correct_keys(self):
        from src.losses.combined_loss import CombinedLoss
        loss_fn = CombinedLoss()
        logits  = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        result  = loss_fn(logits, targets)
        assert isinstance(result, dict)
        for key in ("total", "bce", "dice"):
            assert key in result, f"Missing key: {key}"

    def test_total_is_weighted_sum(self):
        from src.losses.combined_loss import CombinedLoss
        loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        logits  = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        result  = loss_fn(logits, targets)
        expected = 0.5 * result["bce"].item() + 0.5 * result["dice"].item()
        assert abs(result["total"].item() - expected) < 1e-5

    def test_loss_is_positive(self):
        from src.losses.combined_loss import CombinedLoss
        loss_fn = CombinedLoss()
        logits  = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        result  = loss_fn(logits, targets)
        assert result["total"].item() > 0


# ===========================================================================
# Stage 05 — Metrics
# ===========================================================================

class TestSegmentationMetrics:
    def test_compute_returns_all_keys(self):
        from src.metrics.segmentation_metrics import SegmentationMetrics
        m = SegmentationMetrics(threshold=0.5)
        logits  = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        m.update(logits, targets)
        result = m.compute()
        for key in ("dice","iou","pixel_accuracy","precision","recall"):
            assert key in result, f"Missing metric: {key}"

    def test_metrics_in_range(self):
        from src.metrics.segmentation_metrics import SegmentationMetrics
        m = SegmentationMetrics(threshold=0.5)
        logits  = torch.randn(4, 1, 64, 64)
        targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
        m.update(logits, targets)
        result = m.compute()
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"Metric {key}={val} out of [0,1]"

    def test_perfect_predictions(self):
        from src.metrics.segmentation_metrics import SegmentationMetrics
        m = SegmentationMetrics(threshold=0.5)
        targets = torch.ones(1, 1, 32, 32)
        logits  = torch.full((1, 1, 32, 32), 10.0)   # all FG, threshold=0.5
        m.update(logits, targets)
        result = m.compute()
        assert result["dice"]  > 0.99
        assert result["iou"]   > 0.99

    def test_reset_clears_state(self):
        from src.metrics.segmentation_metrics import SegmentationMetrics
        m = SegmentationMetrics()
        logits  = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        m.update(logits, targets)
        result_before = m.compute()
        m.reset()
        # After reset, internal accumulators are 0.
        # A second update after reset should give reproducible results.
        m.update(logits, targets)
        result_after = m.compute()
        # Results after reset + same update should match results before reset
        assert abs(result_before["dice"] - result_after["dice"]) < 1e-5
        assert abs(result_before["iou"]  - result_after["iou"])  < 1e-5
