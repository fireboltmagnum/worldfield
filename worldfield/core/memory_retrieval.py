"""Memory Retrieval — fetches sparse subgraph neighborhoods for attended concepts.

Sits between ConceptAttention and ActivationEngine.
For each attended concept, fetches 1-hop neighborhood (and selective 2-hop for hubs).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class RetrievalResult:
    nodes: dict[str, Any] = field(default_factory=dict)
    edges: list[Any] = field(default_factory=list)
    pruned: int = 0


class MemoryRetrieval:
    """Score-based memory retrieval — distance-independent.

    Scores every graph node (up to max_candidates) by:
        retrieval_score = attention_score * relation_strength * activation
    Returns top-N regardless of hop distance.
    """

    def __init__(
        self,
        graph,
        encoder: Callable[[str], np.ndarray],
        max_retrieved: int = 30,
        max_candidates: int = 200,
    ):
        self.graph = graph
        self.encoder = encoder
        self.max_retrieved = max_retrieved
        self.max_candidates = max_candidates

    def fetch(self, attended_concepts: list[str]) -> RetrievalResult:
        if not attended_concepts:
            return RetrievalResult()

        # Get embedding vectors for attended concepts
        attended_vecs = []
        attended_names = set()
        for name in attended_concepts:
            node = self.graph.get_concept(name)
            if node is None:
                continue
            if node.vector is not None:
                attended_vecs.append(node.vector)
                attended_names.add(name)
            else:
                vec = self.encoder(name)
                attended_vecs.append(vec)
                attended_names.add(name)

        if not attended_vecs:
            return RetrievalResult()

        # Score all graph nodes
        scored_nodes: list[tuple[str, float, Any]] = []
        all_items = list(self.graph.nodes.items())

        # Bound the candidate pool
        if len(all_items) > self.max_candidates:
            all_items = random.sample(all_items, self.max_candidates)

        for nid, node in all_items:
            if node.vector is None:
                continue
            name = node.canonical_name

            # attention_score: max cosine sim to any attended concept
            max_sim = 0.0
            for av in attended_vecs:
                if av.shape != node.vector.shape:
                    continue
                sim = float(np.dot(av, node.vector) /
                            (np.linalg.norm(av) * np.linalg.norm(node.vector) + 1e-12))
                if sim > max_sim:
                    max_sim = sim

            # relation_strength: max edge weight from any attended concept
            max_rel = 0.0
            for att_name in attended_names:
                for rel in self.graph.get_relations(att_name):
                    if rel.target_id == nid:
                        max_rel = max(max_rel, rel.confidence)
                # Incoming: edges where attended concept is target and candidate is source
                att_node = self.graph.get_concept(att_name)
                if att_node is not None:
                    for rel in self.graph.edges:
                        if rel.target_id == att_node.id and rel.source_id == nid:
                            max_rel = max(max_rel, rel.confidence)

            activation = 0.5
            score = max_sim * max(0.1, max_rel) * activation
            if score > 0.0:
                scored_nodes.append((name, score, node))

        # Sort and take top-N
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored_nodes[:self.max_retrieved]

        result = RetrievalResult()
        result.nodes = {name: node for name, score, node in top_nodes}
        result.scores = {name: score for name, score, node in top_nodes}
        result.pruned = max(0, len(scored_nodes) - self.max_retrieved)

        # Fetch edges between retrieved nodes
        retrieved_ids = {node.id for node in result.nodes.values()}
        seen_pairs = set()
        for name in result.nodes:
            for rel in self.graph.get_relations(name):
                if rel.target_id in retrieved_ids:
                    pair = (rel.source_id, rel.target_id)
                    if pair not in seen_pairs:
                        result.edges.append(rel)
                        seen_pairs.add(pair)
            # Incoming edges via reverse adjacency
            node_obj = result.nodes[name]
            if node_obj.id in self.graph.adjacency:
                for pred, eids in self.graph.adjacency[node_obj.id].items():
                    if pred.startswith("~"):
                        for eid in eids:
                            rel = self.graph.edges[eid]
                            if rel.source_id in retrieved_ids:
                                pair = (rel.source_id, rel.target_id)
                                if pair not in seen_pairs:
                                    result.edges.append(rel)
                                    seen_pairs.add(pair)

        return result
