"""Device selection — no torch import at module level."""
from __future__ import annotations


def pick_device():
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
