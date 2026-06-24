"""Image encoder — small CNN projecting to shared latent space.

Architecture ported from day_one (3 conv layers, GroupNorm, GELU, global avg pool).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class ImageEncoder(nn.Module):
    """CNN that maps (B, 3, H, W) images to (B, latent_dim) vectors."""

    def __init__(self, cfg: Config):
        super().__init__()
        w = cfg.cnn_width
        self.net = nn.Sequential(
            nn.Conv2d(cfg.img_channels, w, 3, stride=2, padding=1),
            nn.GroupNorm(8, w),
            nn.GELU(),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, w * 2),
            nn.GELU(),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, w * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(w * 4, cfg.latent_dim)
        self._dim = cfg.latent_dim

    def forward(self, x):
        h = self.net(x)
        return F.normalize(self.proj(h), dim=-1)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy().astype(np.float32)

    def encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        return self.encode(batch)


def load_image_encoder(ckpt_path: str, cfg: Config, device: str = "cpu") -> ImageEncoder:
    """Load a pretrained ImageEncoder from a day_one checkpoint.

    The checkpoint is a dict with key "model" containing state_dict entries
    prefixed by "img_enc.".
    """
    import warnings
    enc = ImageEncoder(cfg)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = raw["model"] if "model" in raw else raw
    filtered = {k.removeprefix("img_enc."): v
                for k, v in state.items() if k.startswith("img_enc.")}
    missing, _ = enc.load_state_dict(filtered, strict=False)
    if missing:
        warnings.warn(f"Missing keys in ImageEncoder: {missing}")
    enc.to(device)
    enc.eval()
    return enc
