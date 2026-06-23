"""Day 3 — persistent world state (plan §5, §6, §21-Layer-1).

The dumbest possible persistent state: ONE vector, updated by EMA as new inputs
arrive. No graph, no attractors, no propagation. The question is whether a
single vector can hold MULTIPLE things seen over time, or whether each update
overwrites the last (the retrieval-wants-specialization vs memory-wants-
compression conflict).
"""
import numpy as np


class WorldStateEMA:
    """world_state <- decay * world_state + (1-decay) * evidence."""
    def __init__(self, dim, decay=0.9):
        self.dim = dim
        self.decay = decay
        self.state = np.zeros(dim, dtype=np.float32)
        self._initialized = False

    def update(self, evidence):
        if not self._initialized:        # first evidence seeds the state directly,
            self.state = evidence.copy() # otherwise step-1 is 90% zeros (a bias)
            self._initialized = True
        else:
            self.state = self.decay * self.state + (1 - self.decay) * evidence
        return self.state


class WorldStateLastOnly:
    """Baseline FLOOR: keep only the most recent evidence. The EMA must beat
    this, or 'memory' is just echoing the latest input."""
    def __init__(self, dim):
        self.dim = dim
        self.state = np.zeros(dim, dtype=np.float32)

    def update(self, evidence):
        self.state = evidence.copy()
        return self.state


class WorldStateConcat:
    """Baseline CEILING: never compress — average ALL evidence seen so far with
    equal weight. Shows what's achievable without forgetting (but doesn't scale
    and has no recency, so it's a reference, not a proposal)."""
    def __init__(self, dim):
        self.dim = dim
        self._sum = np.zeros(dim, dtype=np.float32)
        self._n = 0

    def update(self, evidence):
        self._sum += evidence
        self._n += 1
        self.state = self._sum / self._n
        return self.state
