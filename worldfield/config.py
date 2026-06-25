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

    # Activation layer
    activation_decay: float = 0.3
    activation_spread: float = 0.5
    activation_hops: int = 2

    # Inference engine
    inference_depth: int = 3

    # Language generation
    nlg_backend: str = "template"
    nlg_model: str = "google/flan-t5-small"

    # Continuous learning
    learning_decay: float = 0.995
    learning_penalty: float = 0.5
    learning_prune_conf: float = 0.05
    learning_prune_support: int = 1
    learning_prune_interval: int = 10

    # Context Window
    cw_max_events: int = 20
    cw_max_world_states: int = 10
    cw_max_entities: int = 30
    cw_max_topic_depth: int = 5
    cw_max_references: int = 10
    cw_max_deltas: int = 10
    cw_max_reasoning: int = 10
    cw_max_simulation: int = 5
    cw_max_attention_history: int = 20

    # Concept Attention (hierarchical)
    attention_top_k: int = 15
    attention_max_candidates: int = 50
    attention_goal_similarity_threshold: float = 0.3
    attention_weight_recency: float = 1.0
    attention_weight_relevance: float = 1.0
    attention_weight_goal: float = 1.0
    attention_weight_centrality: float = 0.5
    attention_weight_activation: float = 1.5
    attention_weight_confidence: float = 0.5

    # Concept Attention (hierarchical + recursive)
    attention_passes: int = 3

    # Memory Retrieval (score-based)
    mr_max_retrieved: int = 30
    mr_max_candidates: int = 200

    # Training
    temperature: float = 0.07
