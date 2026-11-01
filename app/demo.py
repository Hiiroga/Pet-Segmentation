"""
app/demo.py
===========
Gradio web interface for Pet Semantic Segmentation.

Run:
    python app/demo.py
    python app/demo.py --checkpoint checkpoints/best_model.pth --port 7860
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy imports (avoid import errors if not installed at definition time)
# ---------------------------------------------------------------------------


def _find_best_checkpoint() -> pathlib.Path | None:
    """Return the best_model.pth or latest epoch checkpoint, if any."""
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    best = ckpt_dir / "best_model.pth"
    if best.exists():
        return best
    epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    return epoch_ckpts[-1] if epoch_ckpts else None


def build_demo(checkpoint_path: str | pathlib.Path | None = None, threshold: float = 0.5):
    """Build and return the Gradio demo interface."""
    import gradio as gr
    from src.inference.predictor import Predictor

    # Load model
    if checkpoint_path is None:
        checkpoint_path = _find_best_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No trained checkpoint found in checkpoints/. "
            "Run python scripts/train.py first."
        )

    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        device_cfg="auto",
        threshold=threshold,
    )

    # ------------------------------------------------------------------
    # Inference function
    # ------------------------------------------------------------------
    def segment(image_input: np.ndarray, threshold_slider: float) -> tuple:
        """Called by Gradio on each submission."""
        if image_input is None:
            return None, None, None, "No image provided."

        predictor.threshold = threshold_slider
        try:
            pil_img = Image.fromarray(image_input.astype(np.uint8))
            binary, prob, overlay = predictor.predict_with_overlay(pil_img)

            # Compute quick metrics
            fg_pixels = (binary == 255).sum()
            total_pixels = binary.size
            fg_ratio = fg_pixels / total_pixels * 100

            info_text = (
                f"Threshold: {threshold_slider:.2f}\n"
                f"Foreground pixels: {fg_pixels:,} / {total_pixels:,} "
                f"({fg_ratio:.1f}%)\n"
                f"Model: {pathlib.Path(str(checkpoint_path)).name}"
            )

            # Convert prob map to displayable image
            prob_display = (prob * 255).astype(np.uint8)

            return binary, prob_display, overlay, info_text

        except Exception as exc:
            err = f"Error during segmentation: {exc}"
            return None, None, None, err

    # ------------------------------------------------------------------
    # Gradio layout
    # ------------------------------------------------------------------
    with gr.Blocks(title="Pet Semantic Segmentation") as demo:
        gr.Markdown(
            """
            # 🐾 Pet Semantic Segmentation
            **Custom U-Net trained from scratch** on the Oxford-IIIT Pet Dataset.

            Upload a cat or dog image to generate a binary foreground/background segmentation mask.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    image_mode="RGB",
                )
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=threshold,
                    step=0.01,
                    label="Segmentation Threshold",
                    info="Sigmoid probability cut-off (default 0.5)",
                )
                run_btn = gr.Button("Segment", variant="primary")

            with gr.Column(scale=2):
                out_mask = gr.Image(label="Binary Mask (white=foreground)", type="numpy")
                out_prob = gr.Image(label="Probability Map", type="numpy")
                out_overlay = gr.Image(label="Overlay (green=foreground)", type="numpy")
                out_info = gr.Textbox(label="Info", lines=4)

        run_btn.click(
            fn=segment,
            inputs=[inp_image, threshold_slider],
            outputs=[out_mask, out_prob, out_overlay, out_info],
        )

        # Examples (if sample images exist)
        sample_dir = PROJECT_ROOT / "outputs" / "dataset_samples"
        example_images = list(sample_dir.glob("*.png"))[:3]
        if example_images:
            gr.Examples(
                examples=[[str(p)] for p in example_images],
                inputs=[inp_image],
            )

        gr.Markdown(
            """
            ---
            **Label convention:** `1=Foreground` (pet) · `2=Background` · `3=Boundary` (ignored)  
            **Architecture:** Custom U-Net (no pretrained weights) · **Loss:** BCE + Dice  
            **Training:** 20 epochs · Adam optimizer · 256×256 input
            """
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Pet Segmentation Gradio demo.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth). Defaults to checkpoints/best_model.pth.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Initial sigmoid threshold (default 0.5).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve on (default 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        dest="no_browser",
        help="Do not open a browser tab automatically (useful for testing).",
    )
    args = parser.parse_args()

    demo = build_demo(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
