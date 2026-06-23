"""Synthetic paired (image, text) dataset.

We draw a colored shape on a small canvas and pair it with a text description.
Because we generate it ourselves, we know exactly what is supposed to align:
this makes the core-claim test honest (plan §12: "build your own small dataset
where you know exactly what is supposed to align").
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset

COLOR_RGB = {
    "red": (220, 40, 40),
    "crimson": (200, 20, 60),     # near red — confusable on purpose
    "green": (40, 200, 80),
    "olive": (110, 130, 30),
    "blue": (50, 90, 230),
    "navy": (20, 30, 120),        # near blue
    "royalblue": (40, 70, 200),   # near blue
    "skyblue": (110, 170, 235),   # near blue
    "yellow": (240, 210, 40),
    "purple": (160, 60, 200),
    "magenta": (220, 40, 200),    # near purple
    "teal": (30, 160, 160),
}


def _draw_shape(img_size, color_rgb, shape, rng):
    """Return HxWx3 float array in [0,1]. Slight jitter so it's not trivial."""
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    c = np.array(color_rgb, dtype=np.float32) / 255.0

    # random center + radius with margins, so position/size vary across samples
    margin = img_size // 5
    r = rng.randint(img_size // 5, img_size // 3)
    cx = rng.randint(margin + r, img_size - margin - r) if img_size - margin - r > margin + r else img_size // 2
    cy = rng.randint(margin + r, img_size - margin - r) if img_size - margin - r > margin + r else img_size // 2

    ys, xs = np.mgrid[0:img_size, 0:img_size]
    if shape == "circle":
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    elif shape == "square":
        mask = (np.abs(xs - cx) <= r) & (np.abs(ys - cy) <= r)
    elif shape == "triangle":
        # upward triangle: inside if below apex and above base within slanted sides
        dy = ys - (cy - r)            # 0 at apex, grows downward
        half = (dy / (2 * r)) * r     # half-width grows linearly with depth
        mask = (dy >= 0) & (dy <= 2 * r) & (np.abs(xs - cx) <= half)
    elif shape == "diamond":
        mask = (np.abs(xs - cx) + np.abs(ys - cy)) <= r   # L1 ball
    elif shape == "pentagon":
        # approximate: inside circle AND above a flat-ish bottom cut
        circ = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        mask = circ & (ys <= cy + int(0.7 * r))
    else:
        raise ValueError(shape)

    img[mask] = c
    # mild noise so reconstruction/contrastive aren't memorizing exact pixels
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


# small set of phrasings so the text encoder can't just match one fixed string
_TEMPLATES = ["a {c} {s}", "the {c} {s}", "{c} {s}", "this is a {c} {s}"]


def _make_text(color, shape, rng):
    return rng.choice(_TEMPLATES).format(c=color, s=shape)


class CharVocab:
    """Tiny char-level vocab. 0 = pad."""
    def __init__(self):
        chars = sorted(set("abcdefghijklmnopqrstuvwxyz "))
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.pad = 0
        self.size = len(chars) + 1

    def encode(self, text, max_len):
        ids = [self.stoi.get(ch, self.pad) for ch in text.lower()][:max_len]
        ids += [self.pad] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class ShapesDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.vocab = CharVocab()
        classes = [(c, s) for c in cfg.colors for s in cfg.shapes]
        self.classes = classes
        self.class_to_idx = {cs: i for i, cs in enumerate(classes)}

        rng = np.random.RandomState(cfg.seed + (0 if split == "train" else 999))
        pyrng = random.Random(cfg.seed + (0 if split == "train" else 999))

        n_per = cfg.samples_per_class
        n_val = int(n_per * cfg.val_frac)
        take = range(n_per - n_val) if split == "train" else range(n_per - n_val, n_per)

        self.items = []
        for (color, shape) in classes:
            for _ in take:
                img = _draw_shape(cfg.img_size, COLOR_RGB[color], shape, rng)
                text = _make_text(color, shape, pyrng)
                self.items.append((img, text, self.class_to_idx[(color, shape)]))
        pyrng.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, text, label = self.items[i]
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        txt_t = self.vocab.encode(text, self.cfg.max_text_len)
        return img_t, txt_t, label
