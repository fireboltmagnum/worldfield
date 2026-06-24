"""Video encoder — sample frames, encode per-frame with ImageEncoder, temporal pool.

Since we don't have a dedicated video model, we reuse the ImageEncoder on
sampled frames and aggregate via mean pooling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from .image import ImageEncoder


class VideoEncoder:
    """Encodes videos by sampling frames and mean-pooling per-frame latents.

    Usage:
        enc = VideoEncoder(image_encoder, cfg)
        vec = enc.encode("path/to/video.mp4")       # (128,)
        vecs = enc.encode_batch(["vid1.mp4", ...])  # (N, 128)
    """

    def __init__(self, image_encoder: ImageEncoder, cfg: Config, device: str = "cpu"):
        self.img_enc = image_encoder
        self.n_frames = cfg.video_frames
        self.sample_rate = cfg.video_sample_rate
        self.img_size = cfg.img_size
        self._dim = cfg.latent_dim
        self._device = device

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, video_path: str) -> np.ndarray:
        frames = self._sample_frames(video_path)
        return self._encode_frames(frames)

    def encode_batch(self, video_paths: list[str]) -> np.ndarray:
        results = []
        for path in video_paths:
            results.append(self.encode(path))
        return np.stack(results, axis=0)

    def _sample_frames(self, video_path: str) -> torch.Tensor:
        """Sample frames from a video file. Returns (n_frames, 3, H, W) tensor."""
        try:
            import av
        except ImportError:
            raise ImportError("pip install av  # PyAV for video decoding")

        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames
        if self.sample_rate == "uniform":
            indices = np.linspace(0, max(total - 1, 0), self.n_frames, dtype=int)
        else:
            indices = range(min(self.n_frames, total))

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_image().resize((self.img_size, self.img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                arr = torch.from_numpy(arr).permute(2, 0, 1)
                frames.append(arr)
            if len(frames) >= self.n_frames:
                break

        container.close()

        if len(frames) < self.n_frames:
            pad = frames[-1:] * (self.n_frames - len(frames))
            frames.extend(pad)

        return torch.stack(frames, dim=0).to(self._device)

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> np.ndarray:
        """frames: (T, 3, H, W) -> (latent_dim,) after mean pool."""
        latents = self.img_enc(frames)
        pooled = latents.mean(dim=0)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.cpu().numpy().astype(np.float32)
