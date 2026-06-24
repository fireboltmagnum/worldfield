"""Text encoder using SentenceTransformer with a learned projection to latent space.

Latent-space alignment with the image encoder is assumed to happen via a
contrastive fine-tuning step (see scripts/align_encoders.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoderProjection(nn.Module):
    """Small learned projection from sentence-transformer space -> latent space."""

    def __init__(self, st_dim: int = 384, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(st_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextEncoder:
    """Wraps a SentenceTransformer model + optional learned projection.

    Until the projection is trained via contrastive alignment, encoding
    returns raw sentence-transformer embeddings (384-dim).  The projection
    is kept for forward compat but not applied by default.

    Usage:
        enc = TextEncoder(latent_dim=128)
        vec = enc.encode("a red square")       # (384,) float32
        vecs = enc.encode_batch(["a", "b"])     # (2, 384)
    """

    def __init__(self, latent_dim: int = 128, model_name: str = "all-MiniLM-L6-v2",
                 device: str | None = None,
                 use_projection: bool = False):
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.use_projection = use_projection

        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(model_name, device=device or "cpu")

        st_dim = self.st.get_sentence_embedding_dimension()
        device = device or "cpu"
        self.proj = TextEncoderProjection(st_dim, latent_dim)
        self.proj.to(device)
        self.proj.eval()
        self._device = device

    @property
    def dim(self) -> int:
        if self.use_projection:
            return self.latent_dim
        return self.st.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        emb = self.st.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        if self.use_projection:
            latent = self.proj(emb.unsqueeze(0))
            return latent.squeeze(0).cpu().numpy().astype(np.float32)
        return emb.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        embs = self.st.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        if self.use_projection:
            latent = self.proj(embs)
            return latent.cpu().numpy().astype(np.float32)
        return embs.cpu().numpy().astype(np.float32)

    def state_dict(self):
        return self.proj.state_dict()

    def load_state_dict(self, sd):
        self.proj.load_state_dict(sd)
