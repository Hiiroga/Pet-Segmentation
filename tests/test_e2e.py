"""
tests/test_e2e.py
=================
Stage 09 – End-to-end tests for the Gradio demo application.

These tests use Playwright to launch the Gradio server, upload a test image,
and verify that a segmentation mask is returned.

Requirements
------------
- Training must be complete (best_model.pth must exist).
- playwright must be installed: `pip install playwright` + `playwright install chromium`

Run
---
    python -m pytest tests/test_e2e.py -v --tb=short

The tests are automatically skipped if:
  - best_model.pth is missing
  - playwright is not installed
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

CHECKPOINT   = _PROJECT_ROOT / "checkpoints" / "best_model.pth"
DEMO_SCRIPT  = _PROJECT_ROOT / "app" / "demo.py"
GRADIO_PORT  = 7861           # use a non-default port to avoid conflicts
GRADIO_URL   = f"http://127.0.0.1:{GRADIO_PORT}"
SERVER_WAIT  = 20             # seconds to wait for Gradio to start


# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _check_playwright() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _check_checkpoint() -> bool:
    return CHECKPOINT.exists()


pytestmark = pytest.mark.skipif(
    not _check_playwright(),
    reason="playwright not installed – run: pip install playwright && playwright install chromium",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gradio_server():
    """Launch the Gradio demo in a subprocess, yield the URL, then terminate."""
    if not _check_checkpoint():
        pytest.skip("best_model.pth not found – run training first.")

    cmd = [
        sys.executable, str(DEMO_SCRIPT),
        "--checkpoint", str(CHECKPOINT),
        "--port", str(GRADIO_PORT),
        "--no-browser",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(_PROJECT_ROOT),
    )

    # Wait for Gradio to be ready
    import urllib.request
    deadline = time.time() + SERVER_WAIT
    ready = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(GRADIO_URL, timeout=2):
                ready = True
                break
        except Exception:
            time.sleep(1)

    if not ready:
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        pytest.fail(
            f"Gradio server did not start within {SERVER_WAIT}s.\n"
            f"STDOUT: {out.decode(errors='replace')}\n"
            f"STDERR: {err.decode(errors='replace')}"
        )

    yield GRADIO_URL

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def sample_image_path(tmp_path_factory) -> Path:
    """Create a small JPEG test image on disk."""
    tmp = tmp_path_factory.mktemp("e2e_imgs")
    img_path = tmp / "test_pet.jpg"
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestGradioDemo:
    def test_server_responds(self, gradio_server):
        """Gradio root URL should return HTTP 200."""
        import urllib.request
        with urllib.request.urlopen(gradio_server, timeout=5) as resp:
            assert resp.status == 200

    def test_upload_image_and_get_mask(self, gradio_server, sample_image_path):
        """
        Upload an image through the Gradio UI and verify a mask image is returned.
        Uses Playwright to drive the browser.
        """
        from playwright.sync_api import sync_playwright, expect

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page    = browser.new_page()
            page.goto(gradio_server, timeout=15_000)

            # Wait for the Gradio interface to be rendered
            page.wait_for_load_state("networkidle", timeout=15_000)

            # Upload image via file input (Gradio renders an <input type="file">)
            file_inputs = page.locator("input[type='file']")
            file_inputs.first.set_input_files(str(sample_image_path))

            # Click the Submit / Predict button
            submit_btn = page.get_by_role("button", name="Segment")
            submit_btn.click(timeout=5_000)

            # Wait for output image to appear (Gradio output component)
            output_img = page.locator(".output-image img, .gradio-image img").first
            expect(output_img).to_be_visible(timeout=60_000)

            # Verify the src attribute is a valid image URL
            src = output_img.get_attribute("src")
            assert src and len(src) > 0, "Output image has no src attribute."

            browser.close()

    def test_api_predict_endpoint(self, gradio_server, sample_image_path, tmp_path):
        """
        Test the Gradio REST API directly (no browser needed).
        Gradio exposes a /run/predict endpoint.
        """
        import base64
        import json
        import urllib.request

        # Encode image as base64
        with open(sample_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = json.dumps({
            "data": [f"data:image/jpeg;base64,{img_b64}"]
        }).encode()

        req = urllib.request.Request(
            f"{gradio_server}/run/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
            # Gradio returns {"data": [...], "duration": ...}
            assert "data" in body
            assert len(body["data"]) > 0
