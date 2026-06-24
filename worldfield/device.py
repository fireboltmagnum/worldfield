"""Device selection — no torch import at module level."""
from __future__ import annotations

import os


def pick_device():
    if os.name == "nt":
        return "cpu"
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
