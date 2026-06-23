"""Central config for the day-one experiment. Keep everything small and visible."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- shared latent space (plan §4: start small so you can observe it) ---
    latent_dim: int = 128

    # --- data ---
    img_size: int = 32          # tiny canvas
    colors: tuple = ("red", "green", "blue", "yellow", "purple")
    shapes: tuple = ("circle", "square", "triangle")
    samples_per_class: int = 400  # generated synthetically, so "free"
    val_frac: float = 0.2

    # --- text encoder (char-level, no tokenizer dependence — plan §7) ---
    max_text_len: int = 24
    char_emb_dim: int = 32
    text_hidden: int = 128

    # --- image encoder ---
    img_channels: int = 3
    cnn_width: int = 32

    # --- training ---
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4
    temperature: float = 0.07   # InfoNCE temperature
    recon_weight: float = 1.0   # weight on reconstruction vs contrastive
    seed: int = 0

    @property
    def num_classes(self) -> int:
        return len(self.colors) * len(self.shapes)
