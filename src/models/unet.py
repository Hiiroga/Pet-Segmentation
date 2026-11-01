"""
src/models/unet.py
==================
U-Net implementation from scratch using pure PyTorch nn.Module.

Architecture summary
--------------------
Encoder (4 stages):
  Stage 1: DoubleConv(3→64)   → MaxPool2d(2)
  Stage 2: DoubleConv(64→128) → MaxPool2d(2)
  Stage 3: DoubleConv(128→256)→ MaxPool2d(2)
  Stage 4: DoubleConv(256→512)→ MaxPool2d(2)

Bottleneck:
  DoubleConv(512→1024)

Decoder (4 stages, bilinear upsample + skip concat + DoubleConv):
  Stage 1: up(1024)→512 concat skip4(512) → DoubleConv(1024→512)
  Stage 2: up(512) →256 concat skip3(256) → DoubleConv(512→256)
  Stage 3: up(256) →128 concat skip2(128) → DoubleConv(256→128)
  Stage 4: up(128) →64  concat skip1(64)  → DoubleConv(128→64)

Output head:
  Conv2d(64, 1, kernel_size=1)  — raw logits (no sigmoid)

For 256×256 input the output is 256×256 single-channel logit map.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks (with padding=1)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Full U-Net for binary segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 3 for RGB).
    out_channels : int
        Number of output channels (default 1 for binary mask).
    base_features : sequence of int
        Channel sizes for the four encoder stages.
        Default: (64, 128, 256, 512).
    """

    def __init__(
        self,
        in_channels:   int = 3,
        out_channels:  int = 1,
        base_features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()

        feats = list(base_features)   # [64, 128, 256, 512]
        bottle = feats[-1] * 2        # 1024

        # ── Encoder ───────────────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, feats[0])
        self.enc2 = DoubleConv(feats[0],    feats[1])
        self.enc3 = DoubleConv(feats[1],    feats[2])
        self.enc4 = DoubleConv(feats[2],    feats[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = DoubleConv(feats[3], bottle)

        # ── Decoder ───────────────────────────────────────────────────
        # After bilinear upsample the feature map halves spatially.
        # Concatenating with the skip produces doubled channels.
        self.dec4 = DoubleConv(bottle   + feats[3], feats[3])  # 1024+512→512
        self.dec3 = DoubleConv(feats[3] + feats[2], feats[2])  # 512+256→256
        self.dec2 = DoubleConv(feats[2] + feats[1], feats[1])  # 256+128→128
        self.dec1 = DoubleConv(feats[1] + feats[0], feats[0])  # 128+64→64

        # ── Output head ───────────────────────────────────────────────
        self.out_conv = nn.Conv2d(feats[0], out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 3, H, W]

        Returns
        -------
        torch.Tensor  shape [B, 1, H, W]  raw logits (no sigmoid)
        """
        # ── Encoder path ──────────────────────────────────────────────
        s1 = self.enc1(x)                   # [B, 64, H, W]
        s2 = self.enc2(self.pool(s1))       # [B, 128, H/2, W/2]
        s3 = self.enc3(self.pool(s2))       # [B, 256, H/4, W/4]
        s4 = self.enc4(self.pool(s3))       # [B, 512, H/8, W/8]

        # ── Bottleneck ────────────────────────────────────────────────
        b  = self.bottleneck(self.pool(s4)) # [B, 1024, H/16, W/16]

        # ── Decoder path ──────────────────────────────────────────────
        d4 = self._up_cat(b,  s4)          # [B, 1024+512, H/8, W/8]
        d4 = self.dec4(d4)                 # [B, 512, H/8, W/8]

        d3 = self._up_cat(d4, s3)          # [B, 512+256, H/4, W/4]
        d3 = self.dec3(d3)                 # [B, 256, H/4, W/4]

        d2 = self._up_cat(d3, s2)          # [B, 256+128, H/2, W/2]
        d2 = self.dec2(d2)                 # [B, 128, H/2, W/2]

        d1 = self._up_cat(d2, s1)          # [B, 128+64, H, W]
        d1 = self.dec1(d1)                 # [B, 64, H, W]

        return self.out_conv(d1)           # [B, 1, H, W]

    # ------------------------------------------------------------------
    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample x to match skip's spatial size, then concat."""
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([x, skip], dim=1)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNet()
    x     = torch.randn(2, 3, 256, 256)
    out   = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {model.count_parameters():,}")
