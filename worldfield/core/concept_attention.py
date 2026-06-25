"""Concept Attention — hierarchical scoring and selection of concepts.

Pipeline: Goal filter → Context filter → World State filter → Scoring
→ Top-K selection + suppression.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ScoredConcept:
    name: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class AttentionResult:
    attended: list[ScoredConcept]
    suppressed: list[ScoredConcept]
    weights: dict[str, float]
    task_mode: str
    n_candidates: int


class ConceptAttention:
    """Hierarchical concept attention.

    Levels:
    1. Goal filter — candidate concepts similar to active goals
    2. Context filter — intersect with context entities + neighbors
    3. World state filter — intersect with recent world state concepts + neighbors
    4. Full scoring — compute signal vector and rank

    Each level degrades gracefully: empty output → fall back to previous level.
    """

    def __init__(
        self,
        graph,
        encoder: Callable[[str], np.ndarray],
        top_k: int = 15,
        max_candidates: int = 50,
        goal_similarity_threshold: float = 0.3,
        passes: int = 3,
        weight_recency: float = 1.0,
        weight_relevance: float = 1.0,
        weight_goal: float = 1.0,
        weight_centrality: float = 0.5,
        weight_activation: float = 1.5,
        weight_confidence: float = 0.5,
    ):
        self.graph = graph
        self.encoder = encoder
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.goal_similarity_threshold = goal_similarity_threshold
        self.passes = passes
        self.weights = {
            "recency": weight_recency,
            "relevance": weight_relevance,
            "goal_alignment": weight_goal,
            "centrality": weight_centrality,
            "activation": weight_activation,
            "confidence": weight_confidence,
        }

    def hierarchical_attend(
        self,
        active_goals: list,
        context_window,
        current_concepts: list[str],
    ) -> AttentionResult:
        candidates: set[str] = set()

        # Level 1: Goal filter
        if active_goals:
            goal_candidates = self._filter_by_goals(active_goals)
            candidates = goal_candidates

        # Level 2: Context filter
        ctx_entities = set(context_window.active_entities.keys())
        ctx_neighbors = set()
        for ent in ctx_entities:
            for neighbor in self._get_neighbors(ent, hops=1):
                ctx_neighbors.add(neighbor)
        all_context = ctx_entities | ctx_neighbors

        if candidates:
            candidates |= all_context
        if not candidates:
            candidates = all_context

        # Level 3: World state filter
        ws_concepts: set[str] = set()
        for snap in context_window.recent_world_states:
            ws_concepts.update(snap.concepts.keys())
        ws_neighbors = set()
        for c in ws_concepts:
            for neighbor in self._get_neighbors(c, hops=1):
                ws_neighbors.add(neighbor)
        all_ws = ws_concepts | ws_neighbors

        if candidates:
            candidates |= all_ws
        if not candidates:
            candidates = all_ws | ctx_entities

        # Also include current input concepts
        candidates.update(current_concepts)

        # Bounded
        if len(candidates) > self.max_candidates:
            candidates = set(list(candidates)[:self.max_candidates])

        # Recursive passes: each pass adds neighbors of the previous attended set
        for pass_idx in range(self.passes):
            if pass_idx > 0 and attended_set:
                # Add 1-hop neighbors of previously attended concepts
                for name in attended_set:
                    for neighbor in self._get_neighbors(name, hops=1):
                        candidates.add(neighbor)
                # Re-bounded
                if len(candidates) > self.max_candidates:
                    candidates = set(list(candidates)[:self.max_candidates])

            # Level 4: Score all candidates
            if not candidates:
                return AttentionResult(
                    attended=[], suppressed=[], weights=dict(self.weights),
                    task_mode="no_candidates", n_candidates=0,
                )

            task_mode, modulated = self._compute_task_mode(active_goals)
            scored = self._score_candidates(
                candidates, active_goals, context_window, modulated,
            )

            # Sort by final score descending
            scored.sort(key=lambda s: s.score, reverse=True)
            attended_set = set(s.name for s in scored[:self.top_k])

        attended = scored[:self.top_k]
        suppressed = scored[self.top_k:]

        return AttentionResult(
            attended=attended,
            suppressed=suppressed,
            weights=modulated,
            task_mode=task_mode,
            n_candidates=len(candidates),
        )

    def _filter_by_goals(self, active_goals: list) -> set[str]:
        candidates: set[str] = set()
        if not self.graph.nodes:
            return candidates

        for goal in active_goals:
            if goal.embedding is None:
                continue
            for node in self.graph.nodes.values():
                if node.vector is None:
                    continue
                sim = float(np.dot(goal.embedding, node.vector) /
                            (np.linalg.norm(goal.embedding) * np.linalg.norm(node.vector) + 1e-12))
                if sim > self.goal_similarity_threshold:
                    candidates.add(node.canonical_name)
        return candidates

    def _get_neighbors(self, concept_name: str, hops: int = 1) -> set[str]:
        neighbors: set[str] = set()
        result = self.graph.query(concept_name, hops=hops)
        for name, entries in result.items():
            for entry in entries:
                if isinstance(entry, dict):
                    c = entry.get("concept")
                    if c:
                        neighbors.add(c)
                neighbors.add(name)
        return neighbors

    def _compute_task_mode(self, active_goals: list) -> tuple[str, dict[str, float]]:
        w = dict(self.weights)
        if not active_goals:
            # Default browsing mode
            w["relevance"] *= 2.0
            w["recency"] *= 2.0
            return "browsing", w

        descs = [g.description.lower() for g in active_goals if hasattr(g, "description")]
        combined = " ".join(descs)

        if any(kw in combined for kw in ("build", "design", "implement")):
            w["goal_alignment"] *= 2.0
            return "construction", w
        if any(kw in combined for kw in ("compare", "contrast")):
            w["relevance"] *= 1.5
            w["centrality"] *= 1.5
            return "comparison", w
        if any(kw in combined for kw in ("find", "locate", "where")):
            w["centrality"] *= 2.0
            return "search", w
        if any(kw in combined for kw in ("plan", "how")):
            w["activation"] *= 1.5
            w["goal_alignment"] *= 1.5
            return "planning", w

        return "goal_driven", w

    def _score_candidates(
        self,
        candidates: set[str],
        active_goals: list,
        context_window,
        weights: dict[str, float],
    ) -> list[ScoredConcept]:
        # Pre-compute goal embeddings (mean vector)
        goal_emb = None
        goal_descriptions = [g.description for g in active_goals if hasattr(g, "description")]
        if goal_descriptions:
            embs = []
            for g in active_goals:
                if hasattr(g, "embedding") and g.embedding is not None:
                    embs.append(g.embedding)
            if embs:
                goal_emb = np.mean(embs, axis=0)

        # Pre-compute topic embedding
        topic_text = ""
        if context_window.topic_stack:
            topic_text = context_window.topic_stack[-1].topic
        topic_emb = self.encoder(topic_text) if topic_text else None

        scored = []
        entities = context_window.active_entities

        for name in candidates:
            node = self.graph.get_concept(name)
            if node is None:
                continue

            signals: dict[str, float] = {}

            # Recency
            if name in entities:
                turns_since = context_window.turn_counter - entities[name].last_seen_turn
                signals["recency"] = 1.0 / (1.0 + turns_since)
            else:
                signals["recency"] = 0.0

            # Relevance (to topic)
            if topic_emb is not None and node.vector is not None:
                sim = float(np.dot(topic_emb, node.vector) /
                            (np.linalg.norm(topic_emb) * np.linalg.norm(node.vector) + 1e-12))
                signals["relevance"] = max(0.0, sim)
            else:
                signals["relevance"] = 0.0

            # Goal alignment
            if goal_emb is not None and node.vector is not None:
                sim = float(np.dot(goal_emb, node.vector) /
                            (np.linalg.norm(goal_emb) * np.linalg.norm(node.vector) + 1e-12))
                signals["goal_alignment"] = max(0.0, sim)
            else:
                signals["goal_alignment"] = 0.0

            # Centrality
            deg = len(self.graph.get_relations(name)) + len(self.graph.get_incoming(name))
            max_deg = max(
                (len(self.graph.get_relations(n)) + len(self.graph.get_incoming(n))
                 for n in [c for c in candidates]),
                default=1,
            )
            signals["centrality"] = deg / max_deg if max_deg > 0 else 0.0

            # Activation (not available at attention time — default 0.5)
            signals["activation"] = 0.5

            # Confidence
            signals["confidence"] = getattr(node, "confidence", 0.5) or 0.5

            # Weighted arithmetic mean
            total_weight = sum(weights.get(k, 1.0) for k in signals)
            total_score = sum(signals[k] * weights.get(k, 1.0) for k in signals)
            final_score = total_score / total_weight if total_weight > 0 else 0.0

            scored.append(ScoredConcept(name=name, score=final_score, signals=signals))

        return scored
