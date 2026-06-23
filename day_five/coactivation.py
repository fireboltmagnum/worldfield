"""Day 5 — emergent relations via co-activation (plan §15: the landscape where
nearby fragments influence each other; repeated evidence strengthens a region).

The substrate stores independent fragments. A RELATION is not stored as a fact;
it EMERGES: when two concepts are activated together within one event, we
strengthen an edge between them (Hebbian — fire together, wire together).

Reasoning = walk these learned edges: retrieve -> activate -> propagate to
neighbors (and neighbors-of-neighbors) -> score candidates. The test is whether
an UNQUERIED associate surfaces, recovered from co-activation history rather than
from a directly stored fact.
"""
import numpy as np


class CoActivationGraph:
    """Weighted graph over concept ids. Edges grow with co-activation."""
    def __init__(self, n_concepts):
        self.n = n_concepts
        self.W = np.zeros((n_concepts, n_concepts), dtype=np.float32)

    def observe_event(self, active_ids, weight=1.0):
        """All concepts active in one event get mutually strengthened edges.
        This is the only place relations are formed — from co-occurrence."""
        for i in active_ids:
            for j in active_ids:
                if i != j:
                    self.W[i, j] += weight

    def propagate(self, seed_ids, hops=2, decay=0.5):
        """Spread activation from seeds across edges for `hops` steps.
        Returns an activation score per concept. Neighbors-of-neighbors reachable
        because we iterate; each hop is attenuated by `decay`."""
        act = np.zeros(self.n, dtype=np.float32)
        act[seed_ids] = 1.0
        # row-normalize edges into transition weights (avoid hubs dominating)
        row_sums = self.W.sum(axis=1, keepdims=True)
        T = np.divide(self.W, row_sums, out=np.zeros_like(self.W), where=row_sums > 0)
        frontier = act.copy()
        total = act.copy()
        for _ in range(hops):
            frontier = (frontier @ T) * decay
            total += frontier
        total[seed_ids] = 0.0   # don't recover the thing you queried with
        return total
