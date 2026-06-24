"""Global configuration for the WorldField system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    latent_dim: int = 128
    n_slots: int = 8
    slot_decay: float = 0.5
    merge_threshold: float = 0.6
    graph_min_support: int = 3
    graph_pmi_floor: float = 0.0
    refine_hops: int = 2
    refine_decay: float = 0.5
    refine_damping: float = 0.6
    refine_iters: int = 6
    refine_keep_frac: float = 0.5
    retrieval_k: int = 10
    retrieval_sim_threshold: float = 0.3
    db_path: str = str(Path.home() / ".worldfield" / "db")
    device: str = "auto"

    # Image encoder (from day_one)
    img_channels: int = 3
    img_size: int = 32
    cnn_width: int = 32

    # Text encoder (sentence transformer)
    text_model: str = "all-MiniLM-L6-v2"
    text_max_length: int = 64

    # Video encoder
    video_frames: int = 8
    video_sample_rate: str = "uniform"
