"""Thin modality front-ends + one shared latent space (plan §7: intelligence
lives in the shared space, the encoders are thin signal-to-feature layers).

No fragment store, no retrieval, no refinement loop — those come later in the
plan. This file is only what's needed to test the alignment claim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Tiny CNN stem -> shared latent."""
    def __init__(self, cfg):
        super().__init__()
        w = cfg.cnn_width
        self.net = nn.Sequential(
            nn.Conv2d(cfg.img_channels, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.GELU(),   # 16
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1), nn.GroupNorm(8, w * 2), nn.GELU(),           # 8
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), nn.GroupNorm(8, w * 4), nn.GELU(),       # 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(w * 4, cfg.latent_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class ImageDecoder(nn.Module):
    """Shared latent -> image. Reconstruction proves the latent carries content."""
    def __init__(self, cfg):
        super().__init__()
        w = cfg.cnn_width
        self.fc = nn.Linear(cfg.latent_dim, w * 4 * 4 * 4)
        self.w = w
        self.net = nn.Sequential(
            nn.ConvTranspose2d(w * 4, w * 2, 4, stride=2, padding=1), nn.GroupNorm(8, w * 2), nn.GELU(),  # 8
            nn.ConvTranspose2d(w * 2, w, 4, stride=2, padding=1), nn.GroupNorm(8, w), nn.GELU(),          # 16
            nn.ConvTranspose2d(w, cfg.img_channels, 4, stride=2, padding=1),                              # 32
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.w * 4, 4, 4)
        return self.net(h)


class TextEncoder(nn.Module):
    """Char-level GRU -> shared latent (no tokenizer dependence, plan §7)."""
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.char_emb_dim, padding_idx=0)
        self.gru = nn.GRU(cfg.char_emb_dim, cfg.text_hidden, batch_first=True)
        self.proj = nn.Linear(cfg.text_hidden, cfg.latent_dim)

    def forward(self, ids):
        h = self.emb(ids)
        _, hn = self.gru(h)
        return self.proj(hn.squeeze(0))


class Worldfield(nn.Module):
    """Holds the two encoders + image decoder around one shared latent space."""
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.img_enc = ImageEncoder(cfg)
        self.txt_enc = TextEncoder(cfg, vocab_size)
        self.img_dec = ImageDecoder(cfg)

    def encode_image(self, img):
        return self.img_enc(img)

    def encode_text(self, ids):
        return self.txt_enc(ids)

    def forward(self, img, ids):
        zi = self.encode_image(img)
        zt = self.encode_text(ids)
        recon = self.img_dec(zi)
        return zi, zt, recon


def info_nce(zi, zt, temperature):
    """Symmetric InfoNCE: matching image/text pairs are positives along the
    diagonal. This is the alignment objective (plan §11B)."""
    zi = F.normalize(zi, dim=-1)
    zt = F.normalize(zt, dim=-1)
    logits = zi @ zt.t() / temperature           # [B, B]
    targets = torch.arange(zi.size(0), device=zi.device)
    loss_i = F.cross_entropy(logits, targets)    # image -> text
    loss_t = F.cross_entropy(logits.t(), targets)  # text -> image
    return 0.5 * (loss_i + loss_t)
