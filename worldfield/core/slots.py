"""Slot memory — working memory with multiple slots.

Ported from day_four/slot_memory.py. Fixes the Day 3 problem (one vector can't
hold multiple concepts) by maintaining K parallel slots with merge/claim/evict.
"""
from __future__ import annotations

import numpy as np


def _norm(x):
    return x / (np.linalg.norm(x) + 1e-12)


class SlotMemory:
    """K-slot memory with similarity routing, EMA merge, and LRU eviction.

    Usage:
        mem = SlotMemory(dim=128, n_slots=8)
        mem.update(some_vector)
        active = mem.active_slots()   # (k, dim)
        state = mem.state             # mean of active slots
    """

    def __init__(self, dim: int, n_slots: int = 8, decay: float = 0.5,
                 merge_threshold: float = 0.6):
        self.dim = dim
        self.n_slots = n_slots
        self.decay = decay
        self.merge_threshold = merge_threshold

        self.slots = np.zeros((n_slots, dim), dtype=np.float32)
        self.used = np.zeros(n_slots, dtype=bool)
        self.last_used = np.zeros(n_slots, dtype=np.int64)
        self._clock = 0

    def update(self, evidence: np.ndarray) -> np.ndarray:
        """Route evidence into a slot. Returns the current state vector."""
        e = _norm(evidence)
        used_idx = np.where(self.used)[0]

        if len(used_idx) > 0:
            sims = self.slots[used_idx] @ e
            best_idx = used_idx[int(np.argmax(sims))]
            best_sim = float(np.max(sims))
        else:
            best_idx = -1
            best_sim = -1.0

        if best_sim >= self.merge_threshold:
            target = best_idx
            self.slots[target] = _norm(
                self.decay * self.slots[target] + (1 - self.decay) * e
            )
        else:
            free = np.where(~self.used)[0]
            if len(free) > 0:
                target = free[0]
                self.slots[target] = e
                self.used[target] = True
            else:
                target = int(np.argmin(self.last_used))
                self.slots[target] = e

        self.last_used[target] = self._clock
        self._clock += 1
        return self.state

    @property
    def state(self) -> np.ndarray:
        """Mean of active slots."""
        used = self.slots[self.used]
        if len(used) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return _norm(used.mean(axis=0))

    def active_slots(self) -> np.ndarray:
        """Return all active slot vectors as (k, dim) array."""
        return self.slots[self.used].copy()

    def active_count(self) -> int:
        return int(np.sum(self.used))

    def active_indices(self) -> list[int]:
        return [int(i) for i in np.where(self.used)[0]]

    def reset(self):
        self.slots.fill(0.0)
        self.used.fill(False)
        self.last_used.fill(0)
        self._clock = 0

    def state_dict(self) -> dict:
        return {
            "slots": self.slots,
            "used": self.used,
            "last_used": self.last_used,
            "clock": self._clock,
        }

    def load_state_dict(self, sd: dict):
        self.slots = sd["slots"]
        self.used = sd["used"]
        self.last_used = sd["last_used"]
        self._clock = sd["clock"]
