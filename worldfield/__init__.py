"""WorldField — a cognitive architecture on a single shared latent space."""
from ._version import __version__
from .config import Config
from .device import pick_device

__all__ = ["__version__", "Config", "pick_device"]
