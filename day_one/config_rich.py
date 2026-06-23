"""Richer config for Day 4.5 confirmation: many concepts, including deliberately
confusable near-duplicate colors (blue/navy/royalblue/skyblue) so slot routing
faces the hard case it must survive. Used by train_rich.py."""
from dataclasses import dataclass
from config import Config


@dataclass
class RichConfig(Config):
    # near-duplicate blues + other colors => routing must separate similar concepts
    colors: tuple = ("red", "crimson", "green", "olive", "blue", "navy",
                     "royalblue", "skyblue", "yellow", "purple", "magenta", "teal")
    shapes: tuple = ("circle", "square", "triangle", "diamond", "pentagon")
    samples_per_class: int = 160
    epochs: int = 35
    # 12 colors x 5 shapes = 60 concepts (vs 15 before)
