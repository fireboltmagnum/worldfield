"""Sample data generation for testing WorldField."""
from __future__ import annotations

import numpy as np


SAMPLE_CONCEPTS = [
    "a red square", "a blue circle", "a green triangle",
    "a yellow star", "a purple diamond", "an orange hexagon",
    "a cat", "a dog", "a bird", "a fish",
    "sofa", "table", "chair", "lamp",
    "ocean", "mountain", "forest", "desert",
    "sun", "moon", "star", "cloud",
    "happy", "sad", "angry", "calm",
]


def generate_sample_texts(n: int = 50, seed: int = 42) -> list[str]:
    """Generate sample text inputs from the concept list."""
    rng = np.random.RandomState(seed)
    texts = []
    for _ in range(n):
        concept = rng.choice(SAMPLE_CONCEPTS)
        prefix = rng.choice(["", "a ", "the ", "this is a "])
        texts.append(f"{prefix}{concept}")
    return texts


def generate_related_events(n_events: int = 20, seed: int = 42) -> list[list[str]]:
    """Generate events containing multiple related concepts."""
    rng = np.random.RandomState(seed)
    groups = [
        ["a cat", "sofa", "lamp", "table"],
        ["a dog", "bone", "park", "ball"],
        ["ocean", "fish", "bird", "cloud"],
        ["mountain", "forest", "tree", "river"],
        ["happy", "sun", "star", "calm"],
    ]
    events = []
    for _ in range(n_events):
        group = rng.choice(groups)
        k = rng.randint(2, len(group) + 1)
        events.append(list(rng.choice(group, size=k, replace=False)))
    return events
