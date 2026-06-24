"""Concept module — temporal memory, confidence, uncertainty, hierarchy.

Extends the basic fragment/slot/graph system with higher-level concept tracking.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Concept:
    """A tracked concept with confidence, uncertainty, and temporal state.

    Concepts are higher-level abstractions built from related fragments.
    """
    name: str
    vector: np.ndarray
    first_seen: float = 0.0
    last_seen: float = 0.0
    n_observations: int = 0
    confidence: float = 0.0
    uncertainty: float = 1.0
    parent: str | None = None
    children: list[str] = field(default_factory=list)


class ConceptMemory:
    """Temporal concept memory with confidence tracking and hierarchy.

    Tracks:
    - Temporal decay (concepts weaken if not seen recently)
    - Confidence (based on frequency and consistency)
    - Uncertainty (based on dispersion of fragment vectors)
    - Hierarchy (parent-child relationships)
    """

    def __init__(self, decay_half_life: float = 3600.0):
        self.concepts: dict[str, Concept] = {}
        self.decay_half_life = decay_half_life
        self._next_id = 0
        self._fragment_to_concept: dict[str, str] = {}

    def observe(self, fragment_id: str, vector: np.ndarray,
                label: str | None = None, timestamp: float | None = None) -> str:
        """Add or update a concept from an observation.

        If label is provided, it maps the fragment to an existing concept
        or creates a new one.
        """
        ts = timestamp or time.time()
        name = label or f"concept_{self._next_id}"

        if name not in self.concepts:
            self.concepts[name] = Concept(
                name=name,
                vector=vector.copy(),
                first_seen=ts,
                last_seen=ts,
                n_observations=1,
            )
            self._next_id += 1
        else:
            c = self.concepts[name]
            # EMA update with recency
            alpha = 1.0 / (c.n_observations + 1)
            c.vector = (1 - alpha) * c.vector + alpha * vector
            c.vector /= np.linalg.norm(c.vector) + 1e-12
            c.last_seen = ts
            c.n_observations += 1

        # Update confidence
        self._update_confidence(name)

        # Map fragment to concept
        self._fragment_to_concept[fragment_id] = name
        return name

    def _update_confidence(self, name: str):
        """Confidence = f(frequency, recency)."""
        c = self.concepts[name]
        if c.n_observations == 0:
            c.confidence = 0.0
            c.uncertainty = 1.0
            return

        # Frequency component (saturating)
        freq_conf = 1.0 - 1.0 / (1.0 + c.n_observations * 0.5)

        # Recency component (temporal decay)
        age = time.time() - c.last_seen
        recency = 2.0 ** (-age / self.decay_half_life)

        # Combined confidence
        c.confidence = float(freq_conf * recency)

        # Uncertainty = 1 - confidence (simple model)
        c.uncertainty = float(max(0.01, 1.0 - c.confidence))

    def decay_all(self, timestamp: float | None = None):
        """Apply temporal decay to all concepts."""
        ts = timestamp or time.time()
        for name in list(self.concepts.keys()):
            c = self.concepts[name]
            age = ts - c.last_seen
            if age > self.decay_half_life * 10:
                del self.concepts[name]
            else:
                self._update_confidence(name)

    def get_by_fragment(self, fragment_id: str) -> Concept | None:
        """Get the concept associated with a fragment."""
        name = self._fragment_to_concept.get(fragment_id)
        if name and name in self.concepts:
            return self.concepts[name]
        return None

    def add_hierarchy(self, parent: str, child: str):
        """Add a parent-child relationship between concepts."""
        if parent in self.concepts and child in self.concepts:
            if parent not in self.concepts[child].children:
                self.concepts[child].children.append(parent)
            self.concepts[parent].children.append(child)
            self.concepts[child].parent = parent

    def get_children(self, name: str) -> list[Concept]:
        """Get child concepts of a given concept."""
        if name not in self.concepts:
            return []
        return [self.concepts[c] for c in self.concepts[name].children if c in self.concepts]

    def get_parent(self, name: str) -> Concept | None:
        """Get the parent concept."""
        if name not in self.concepts:
            return None
        p = self.concepts[name].parent
        return self.concepts.get(p)

    def top_concepts(self, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k concepts by confidence."""
        sorted_c = sorted(
            self.concepts.items(),
            key=lambda x: x[1].confidence,
            reverse=True,
        )
        return [(n, c.confidence) for n, c in sorted_c[:k]]

    def state_dict(self) -> dict:
        return {
            "concepts": {
                name: {
                    "name": c.name,
                    "vector": c.vector.tolist(),
                    "first_seen": c.first_seen,
                    "last_seen": c.last_seen,
                    "n_observations": c.n_observations,
                    "confidence": c.confidence,
                    "uncertainty": c.uncertainty,
                    "parent": c.parent,
                    "children": c.children,
                }
                for name, c in self.concepts.items()
            },
            "fragment_to_concept": self._fragment_to_concept,
            "decay_half_life": self.decay_half_life,
        }

    def load_state_dict(self, sd: dict):
        self.concepts = {}
        for name, data in sd["concepts"].items():
            self.concepts[name] = Concept(
                name=data["name"],
                vector=np.array(data["vector"], dtype=np.float32),
                first_seen=data["first_seen"],
                last_seen=data["last_seen"],
                n_observations=data["n_observations"],
                confidence=data["confidence"],
                uncertainty=data["uncertainty"],
                parent=data["parent"],
                children=data["children"],
            )
        self._fragment_to_concept = sd.get("fragment_to_concept", {})
        self.decay_half_life = sd.get("decay_half_life", self.decay_half_life)
