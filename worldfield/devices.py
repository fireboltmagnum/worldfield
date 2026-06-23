"""Device selection kept separate from heavy experiment scripts."""

from __future__ import annotations


def pick_device():
    """Return a torch device, falling back to the string ``cpu`` without torch."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
