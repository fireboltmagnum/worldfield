"""Activation Layer — spreading activation over the concept graph.

When concepts are mentioned, their activation level increases and spreads
to related concepts via graph edges. Activation decays over time (ticks)
so the system maintains :emphasis:`what is relevant right now`.

Usage::

    engine = ActivationEngine(graph)
    engine.trigger(["cat", "sofa"], strength=1.0)
    engine.spread()
    active = engine.get_active(threshold=0.1)
"""
from __future__ import annotations

from typing import Any

from .world_graph import WorldGraph


class ActivationEngine:
    """Manages concept activation with spreading and decay.

    Parameters
    ----------
    graph:
        The :class:`WorldGraph` used to look up concept IDs and edges.
    decay_rate:
        Fraction of activation lost on each :meth:`tick` (0..1).
    spread_factor:
        Fraction of a node's activation distributed to neighbours.
    spread_hops:
        How many edge hops activation spreads.
    working_set_threshold:
        Concepts above this activation enter the persistent working set.
    min_activation:
        Floor below which concepts are treated as inactive.
    """

    def __init__(
        self,
        graph: WorldGraph,
        decay_rate: float = 0.3,
        spread_factor: float = 0.5,
        spread_hops: int = 2,
        working_set_threshold: float = 0.5,
        min_activation: float = 0.01,
    ) -> None:
        self.graph = graph
        self.decay_rate = decay_rate
        self.spread_factor = spread_factor
        self.spread_hops = spread_hops
        self.working_set_threshold = working_set_threshold
        self.min_activation = min_activation

        self._activation: dict[str, float] = {}
        self._working_set: dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def trigger(self, concept_names: list[str], strength: float = 1.0) -> None:
        """Boost activation for directly mentioned concepts.

        Parameters
        ----------
        concept_names:
            Surface forms (resolved via the graph).
        strength:
            Amount of activation added per concept.
        """
        for name in concept_names:
            node = self.graph.get_concept(name)
            if node is None:
                continue
            current = self._activation.get(node.id, 0.0)
            self._activation[node.id] = current + strength

    def spread(self) -> None:
        """Propagate activation to related concepts via graph edges."""
        import copy
        new_activation: dict[str, float] = {}
        for cid, level in self._activation.items():
            if level <= 0.0:
                continue
            retained = level * (1.0 - self.spread_factor)
            new_activation[cid] = new_activation.get(cid, 0.0) + retained

            frontier: list[tuple[str, float, int]] = [
                (cid, level * self.spread_factor, 0)
            ]
            visited: set[str] = {cid}
            while frontier:
                cur_id, energy, depth = frontier.pop(0)
                if depth >= self.spread_hops:
                    continue
                for edge in self._outgoing_edges(cur_id):
                    nid = edge.target_id
                    if nid in visited:
                        continue
                    visited.add(nid)
                    spread = energy * edge.confidence
                    if spread < self.min_activation:
                        continue
                    new_activation[nid] = new_activation.get(nid, 0.0) + spread
                    frontier.append((nid, spread * self.spread_factor, depth + 1))

        self._activation = new_activation

    def tick(self) -> None:
        """Apply time decay. Concepts below *min_activation* are removed.

        Working set is updated *before* decay so it captures activations at
        their peak (before they fade for the next turn).
        """
        self._update_working_set()
        to_remove = [cid for cid in self._activation
                     if (self._activation[cid] * (1.0 - self.decay_rate))
                     < self.min_activation]
        for cid in to_remove:
            del self._activation[cid]
        for cid in self._activation:
            self._activation[cid] *= 1.0 - self.decay_rate

    def get_active(
        self, threshold: float = 0.1
    ) -> list[tuple[str, float]]:
        """Return (concept_name, activation) pairs sorted descending."""
        result: list[tuple[str, float]] = []
        for cid, level in self._activation.items():
            if level >= threshold and cid in self.graph.nodes:
                result.append((self.graph.nodes[cid].canonical_name, level))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_working_set(self, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k persistently activated concepts."""
        sorted_ws = sorted(
            self._working_set.items(), key=lambda x: x[1], reverse=True
        )
        result: list[tuple[str, float]] = []
        for cid, level in sorted_ws[:k]:
            if cid in self.graph.nodes:
                result.append((self.graph.nodes[cid].canonical_name, level))
        return result

    def get_activation_map(self) -> dict[str, float]:
        """Return raw ``{concept_id: activation}`` (for serialisation)."""
        return dict(self._activation)

    def reset(self) -> None:
        """Clear all activation and working-set state."""
        self._activation.clear()
        self._working_set.clear()

    # ── Internal helpers ───────────────────────────────────────────────

    def _outgoing_edges(self, cid: str) -> list[Any]:
        """Yield non-reverse edges whose source is *cid*."""
        edges: list[Any] = []
        adj = self.graph.adjacency
        if cid not in adj:
            return edges
        for pred, eids in adj[cid].items():
            if pred.startswith("~"):
                continue
            for eid in eids:
                edges.append(self.graph.edges[eid])
        return edges

    def _update_working_set(self) -> None:
        """Promote high-activation concepts to the working set."""
        for cid, level in self._activation.items():
            if level >= self.working_set_threshold:
                self._working_set[cid] = max(
                    self._working_set.get(cid, 0.0), level
                )
        to_remove = [
            cid
            for cid in self._working_set
            if self._working_set[cid] * (1.0 - self.decay_rate * 0.5)
            < self.min_activation
        ]
        for cid in to_remove:
            del self._working_set[cid]
        for cid in self._working_set:
            self._working_set[cid] *= 1.0 - self.decay_rate * 0.5

    def state_dict(self) -> dict[str, Any]:
        return {
            "activation": dict(self._activation),
            "working_set": dict(self._working_set),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._activation = dict(sd.get("activation", {}))
        self._working_set = dict(sd.get("working_set", {}))
