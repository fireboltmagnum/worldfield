"""WorldField training — multimodal alignment training.

Phase 1: Populate World Graph from COCO captions + GQA scene graphs
Phase 2: Train text↔image projection alignment
Phase 3: Evaluate concept understanding via graph queries
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..core import Engine, WorldGraph
from ..encoders.text import TextEncoderProjection


class CaptionImageDataset(Dataset):
    """Pairs COCO captions with image paths for contrastive training."""

    def __init__(self, captions: list[dict], image_dir: str, image_size: int = 224):
        self.pairs = []
        for cap in captions:
            img_id = cap["image_id"]
            img_path = Path(image_dir) / f"COCO_train2014_{int(img_id):012d}.jpg"
            if img_path.exists():
                self.pairs.append((cap["caption"], str(img_path), cap["id"]))
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        caption, img_path, ann_id = self.pairs[idx]
        return caption, img_path


def train_projection(
    engine: Engine,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    out_path: str = "out/projection.pt",
) -> dict[str, Any]:
    """Train text encoder projection layer via caption↔image contrastive loss.

    The text encoder's projection is randomly initialized.  This aligns it
    with the (frozen) image encoder so that matching caption-image pairs
    have similar embeddings.
    """
    proj = engine.text_encoder.proj
    proj.to(device)
    proj.train()

    img_enc = engine.image_encoder
    img_enc.to(device)
    img_enc.eval()

    text_enc = engine.text_encoder
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    from PIL import Image

    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    history = {"loss": []}
    from tqdm import tqdm

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for captions, img_paths in pbar:
            # Encode text
            text_vecs = []
            for cap in captions:
                v = text_enc.encode(cap)
                text_vecs.append(v)
            text_embs = torch.tensor(np.stack(text_vecs)).to(device)

            # Encode images
            img_tensors = []
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img_tensors.append(transform(img))
                except Exception:
                    img_tensors.append(torch.zeros(3, 224, 224))
            if not img_tensors:
                continue
            img_batch = torch.stack(img_tensors).to(device)
            with torch.no_grad():
                img_embs = img_enc(img_batch)

            # Normalize
            text_embs = F.normalize(text_embs, dim=-1)
            img_embs = F.normalize(img_embs, dim=-1)

            # Contrastive loss (InfoNCE)
            logits = text_embs @ img_embs.T * engine.cfg.temperature
            labels = torch.arange(len(logits)).to(device)
            loss_t2i = F.cross_entropy(logits, labels)
            loss_i2t = F.cross_entropy(logits.T, labels)
            loss = (loss_t2i + loss_i2t) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)
        print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")

        if val_loader:
            val_loss = _eval_projection(proj, text_enc, img_enc,
                                        val_loader, transform, device)
            print(f"  Val loss = {val_loss:.4f}")

    # Save projection weights
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj.state_dict(), str(out))
    print(f"Projection saved to {out_path}")

    proj.eval()
    engine._text_encoder.use_projection = True
    history["out_path"] = out_path
    return history


def _eval_projection(proj, text_enc, img_enc, loader, transform, device):
    proj.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for captions, img_paths in loader:
            text_vecs = []
            for cap in captions:
                v = text_enc.encode(cap)
                text_vecs.append(v)
            text_embs = torch.tensor(np.stack(text_vecs)).to(device)

            img_tensors = []
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img_tensors.append(transform(img))
                except Exception:
                    img_tensors.append(torch.zeros(3, 224, 224))
            if not img_tensors:
                continue
            img_batch = torch.stack(img_tensors).to(device)
            img_embs = img_enc(img_batch)

            text_embs = F.normalize(text_embs, dim=-1)
            img_embs = F.normalize(img_embs, dim=-1)

            logits = text_embs @ img_embs.T
            labels = torch.arange(len(logits)).to(device)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2
            total_loss += loss.item()
            n += 1
    proj.train()
    return total_loss / max(n, 1)
