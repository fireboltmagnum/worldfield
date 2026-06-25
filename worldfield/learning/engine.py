"""Continuous learning engine — updates long-term memory from every turn.

No epochs, no retraining loops. Every input is a training example.
Confidence updates are immediate, contradictions trigger resolution,
and memory pruning prevents unbounded growth.
"""
from __future__ import annotations

import time
from typing import Any

from ..core.world_graph import WorldGraph
from ..core.graph import PMIGraph


class Resolution:
    """The result of resolving a contradiction between two beliefs."""

    def __init__(
        self,
        winner: str,
        loser: str,
        winner_conf: float,
        loser_original_conf: float,
        loser_new_conf: float,
    ):
        self.winner = winner
        self.loser = loser
        self.winner_conf = winner_conf
        self.loser_original_conf = loser_original_conf
        self.loser_new_conf = loser_new_conf


class LearningEngine:
    """Updates the graph continuously as the system processes input.

    Parameters
    ----------
    graph:
        The :class:`WorldGraph` to update.
    confidence_decay:
        Per-turn multiplicative factor applied to all edge confidences
        (simulating forgetting). 1.0 = no decay.
    contradiction_penalty:
        Fraction of confidence lost by the weaker belief in a contradiction.
    prune_confidence_threshold:
        Edges below this confidence are removed during pruning.
    prune_support_threshold:
        Edges whose ``support_count`` is below this value are removed.
    prune_interval:
        How many turns between pruning passes.
    """

    def __init__(
        self,
        graph: WorldGraph,
        confidence_decay: float = 0.995,
        contradiction_penalty: float = 0.5,
        prune_confidence_threshold: float = 0.05,
        prune_support_threshold: int = 1,
        prune_interval: int = 10,
    ):
        self.graph = graph
        self.pmi: PMIGraph | None = None
        self.confidence_decay = confidence_decay
        self.contradiction_penalty = contradiction_penalty
        self.prune_confidence_threshold = prune_confidence_threshold
        self.prune_support_threshold = prune_support_threshold
        self.prune_interval = prune_interval
        self._turn_count = 0
        self._last_prune_turn = 0
        self._last_update_time = time.time()

    # ── Public API ─────────────────────────────────────────────────────

    def observe(self, turn_result: dict[str, Any]) -> list[Resolution]:
        """Process one pipeline result to update long-term memory.

        Called once per turn, after the engine pipeline completes.
        """
        self._turn_count += 1
        resolutions: list[Resolution] = []

        # 1. Confidence decay (forgetting)
        self._decay_edges()

        # 2. Resolve contradictions from inference result
        inference = turn_result.get("inference_result", {})
        if inference and inference.get("contradictions"):
            resolutions = self._resolve_contradictions(
                inference["contradictions"]
            )

        # 3. Periodic pruning
        if self._turn_count - self._last_prune_turn >= self.prune_interval:
            self.prune()
            self._last_prune_turn = self._turn_count

        self._last_update_time = time.time()
        return resolutions

    def reinforce(
        self,
        concept_names: list[str],
        relation_triples: list[tuple[str, str, str, float]],
    ) -> None:
        """Manually reinforce concepts and relations.

        Normally the graph's own ``relate()`` / ``_touch()`` methods handle
        this during :meth:`process`. This method is for explicit reinforcement
        outside the pipeline.
        """
        now = time.time()
        for name in concept_names:
            node = self.graph.get_concept(name)
            if node is not None:
                node.last_seen = now
                node.activation_count += 1
                node.confidence = 1.0 - 1.0 / (1.0 + node.activation_count * 0.5)

        for src, pred, tgt, conf in relation_triples:
            existing = self._find_edge_by_names(src, pred, tgt)
            if existing is not None:
                existing.support_count += 1
                existing.last_seen = now
                existing.last_confirmed = now
                existing.confidence = 1.0 - 1.0 / (1.0 + existing.support_count * 0.5)
            else:
                self.graph.relate(src, pred, tgt)

    def prune(self) -> int:
        """Remove low-confidence or low-support relations.

        Returns
        -------
        int
            Number of edges removed.
        """
        before = len(self.graph.edges)
        to_remove: list[int] = []
        for i, edge in enumerate(self.graph.edges):
            if edge.confidence < self.prune_confidence_threshold:
                to_remove.append(i)
            elif edge.support_count < self.prune_support_threshold:
                to_remove.append(i)

        if not to_remove:
            return 0

        # Remove in reverse order to preserve indices
        for i in sorted(to_remove, reverse=True):
            del self.graph.edges[i]

        # Rebuild adjacency index
        self.graph.adjacency = {}
        for i, edge in enumerate(self.graph.edges):
            sid = edge.source_id
            tid = edge.target_id
            pred = edge.predicate
            self.graph.adjacency.setdefault(sid, {})
            self.graph.adjacency[sid].setdefault(pred, []).append(i)
            self.graph.adjacency.setdefault(tid, {})
            self.graph.adjacency[tid].setdefault(f"~{pred}", []).append(i)

        removed = before - len(self.graph.edges)
        return removed

    def refine_concepts(self, result: dict[str, Any]) -> None:
        """Update concept aliases and vectors from a pipeline result.

        Called automatically during :meth:`observe`.
        """
        extracted = result.get("extracted_concepts_raw", [])
        for c in extracted:
            name = c.get("name", "")
            vec = c.get("vector")
            node = self.graph.get_concept(name)
            if node is not None and vec is not None:
                if node.vector is None:
                    node.vector = vec.copy()
                else:
                    # Running average with new vector
                    alpha = 0.3
                    node.vector = (1.0 - alpha) * node.vector + alpha * vec

    # ── Internal ──────────────────────────────────────────────────────

    def _decay_edges(self) -> None:
        """Apply multiplicative confidence decay to all edges."""
        for edge in self.graph.edges:
            edge.confidence *= self.confidence_decay

    def _resolve_contradictions(
        self, contradictions: list[dict[str, Any]]
    ) -> list[Resolution]:
        """For each contradiction, reduce the weaker belief's confidence."""
        resolutions: list[Resolution] = []
        for c in contradictions:
            entity = c.get("entity", "")
            target = c.get("target", "")
            pred_a = c.get("predicate_a", "")
            pred_b = c.get("predicate_b", "")
            conf_a = c.get("confidence_a", 0.0)
            conf_b = c.get("confidence_b", 0.0)

            # The stronger belief is the "winner"
            if conf_a >= conf_b:
                winner_pred, loser_pred = pred_a, pred_b
                winner_conf, loser_conf = conf_a, conf_b
            else:
                winner_pred, loser_pred = pred_b, pred_a
                winner_conf, loser_conf = conf_b, conf_a

            # Find and weaken the loser in the graph
            loser_edge = self._find_edge_by_names(entity, loser_pred, target)
            if loser_edge is not None:
                original = loser_edge.confidence
                loser_edge.confidence *= self.contradiction_penalty
                # Reduce support count too
                loser_edge.support_count = max(
                    1, loser_edge.support_count - 1
                )
                resolutions.append(Resolution(
                    winner=f"{entity} -[{winner_pred}]-> {target}",
                    loser=f"{entity} -[{loser_pred}]-> {target}",
                    winner_conf=winner_conf,
                    loser_original_conf=original,
                    loser_new_conf=loser_edge.confidence,
                ))

        return resolutions

    def _find_edge_by_names(
        self, source_name: str, predicate: str, target_name: str
    ):
        """Find a graph edge by canonical names (not IDs)."""
        src = self.graph.get_concept(source_name)
        if src is None:
            return None
        tgt = self.graph.get_concept(target_name)
        if tgt is None:
            return None
        return self.graph._find_edge(src.id, predicate, tgt.id)
