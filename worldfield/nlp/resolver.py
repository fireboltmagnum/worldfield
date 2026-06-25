"""Concept Resolver — resolves surface forms to canonical concepts.

Preserves original forms as aliases. Uses vector similarity and edit distance.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.world_graph import WorldGraph, SIMILARITY_THRESHOLD


class ConceptResolver:
    """Resolves surface forms to concepts in the World Graph.

    Tries, in order:
    1. Exact alias match
    2. Vector similarity match (above SIMILARITY_THRESHOLD)
    3. Close edit distance (< 3 chars, same first letter)
    4. Creates new concept
    """

    def __init__(self, graph: WorldGraph):
        self.graph = graph

    def resolve(self, surface_form: str, vector: np.ndarray | None = None,
                modality: str = "", source: str = "",
                example: str = "") -> str:
        """Resolve a surface form to a concept name. Returns canonical name."""
        node = self.graph.resolve(surface_form, vector, modality, source, example)
        return node.canonical_name

    def extract_and_resolve(self, text: str, concepts: list[dict],
                            relations: list[dict],
                            modality: str = "text", source: str = "") -> tuple[
            list[dict], list[dict]]:
        """Run extraction results through the resolver.

        Returns updated concepts and relations with canonical names.
        """
        resolved_conc = []
        for c in concepts:
            canonical = self.resolve(
                c["name"],
                vector=c.get("vector"),
                modality=modality,
                source=source,
                example=text,
            )
            resolved_conc.append({
                **c,
                "name": canonical,
                "resolved_from": c["name"],
            })

        resolved_rel = []
        for r in relations:
            src = self.resolve(r["source"], modality=modality, source=source, example=text)
            tgt = self.resolve(r["target"], modality=modality, source=source, example=text)
            entry = {**r, "source": src, "target": tgt}
            # Negated relations set polarity to False
            if r.get("negated"):
                entry["polarity"] = False
            resolved_rel.append(entry)

        return resolved_conc, resolved_rel
