"""Day 4 — slot memory (plan §21: layered memory; multiple state vectors).

Day 3 proved one vector can't be read back: averaging N concepts gives a centroid
that's no single concept's sharp point. The fix the plan names is to NOT compress
everything into one point — keep K slots, so each concept can occupy its own.

Routing rule (the dumbest that could work):
  - normalize incoming evidence
  - find the most similar EXISTING (used) slot
  - if its similarity >= merge_threshold -> update that slot (same concept seen again)
  - else if a free slot exists -> claim a free slot (a new concept)
  - else -> overwrite the least-recently-used slot (capacity pressure)

No learning, no attention — just routing + per-slot EMA. The question: does this
recover the items that one vector could not, on the SAME capacity test?
"""
import numpy as np


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


class SlotMemory:
    def __init__(self, dim, n_slots=8, decay=0.5, merge_threshold=0.6):
        self.dim = dim
        self.n_slots = n_slots
        self.decay = decay
        self.merge_threshold = merge_threshold
        self.slots = np.zeros((n_slots, dim), dtype=np.float32)
        self.used = np.zeros(n_slots, dtype=bool)
        self.last_used = np.zeros(n_slots, dtype=np.int64)
        self._clock = 0
        # `state` is exposed for the shared harness: the readable memory is the
        # set of used slots. For metrics we expose the slot matrix directly.

    @property
    def state(self):
        """Backward-compat single-vector view (mean of used slots). The honest
        readout, though, is `active_slots()` — used by the Day-4 metric."""
        if not self.used.any():
            return np.zeros(self.dim, dtype=np.float32)
        return self.slots[self.used].mean(0)

    def active_slots(self):
        return self.slots[self.used]

    def update(self, evidence):
        self._clock += 1
        e = _norm(evidence).astype(np.float32)

        if self.used.any():
            sims = self.slots[self.used] @ e
            used_idx = np.where(self.used)[0]
            best_local = int(np.argmax(sims))
            best = used_idx[best_local]
            best_sim = float(sims[best_local])
        else:
            best, best_sim = None, -1.0

        if best is not None and best_sim >= self.merge_threshold:
            # same concept seen again: refine that slot
            self.slots[best] = self.decay * self.slots[best] + (1 - self.decay) * e
            self.slots[best] = _norm(self.slots[best])
            target = best
        elif not self.used.all():
            # new concept: claim a free slot
            target = int(np.where(~self.used)[0][0])
            self.slots[target] = e
            self.used[target] = True
        else:
            # full: evict least-recently-used (capacity pressure)
            target = int(np.argmin(self.last_used))
            self.slots[target] = e
        self.last_used[target] = self._clock
        return self.state
