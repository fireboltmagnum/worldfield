"""PMI graph — relation learning via pointwise mutual information.

Ports the key algorithms from day_eight (PMI learning) and day_five (propagate/diffuse).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags


class PMIGraph:
    """Learns relations between fragments via PMI/lift and supports propagation.

    Usage:
        g = PMIGraph()
        g.observe([frag_id_1, frag_id_2, ...])
        g.finalize()
        scores = g.propagate(seed_ids, seed_vals)
    """

    def __init__(self, min_support: int = 3, pmi_floor: float = 0.0):
        self.min_support = min_support
        self.pmi_floor = pmi_floor
        self.N = 0
        self.ni: dict[int, int] = {}
        self.nij: dict[tuple[int, int], int] = {}
        self._W: csr_matrix | None = None
        self._T: csr_matrix | None = None

    def observe(self, frag_ids: list[int], weights: list[float] | None = None):
        """Record co-occurrence of a set of fragment IDs."""
        self.N += 1
        unique = sorted(set(frag_ids))
        for fid in unique:
            self.ni[fid] = self.ni.get(fid, 0) + 1
        for a, b in combinations(unique, 2):
            key = (a, b)
            self.nij[key] = self.nij.get(key, 0) + 1

    def finalize(self) -> "PMIGraph":
        """Build the PMI-weighted adjacency matrix from accumulated counts."""
        if not self.ni:
            self._W = None
            self._T = None
            return self
        n_nodes = max(list(self.ni.keys()) + [0]) + 1
        if n_nodes > 100000:
            raise RuntimeError(
                f"PMI graph has {n_nodes} nodes (max ID {n_nodes-1}). "
                "Use compact sequential IDs instead of large hash values."
            )
        W = lil_matrix((n_nodes, n_nodes), dtype=np.float32)

        for (i, j), cij in self.nij.items():
            if cij < self.min_support:
                continue
            pi = self.ni[i] / self.N
            pj = self.ni[j] / self.N
            pij = cij / self.N
            if pi * pj == 0:
                continue
            pmi = math.log(pij / (pi * pj) + 1e-12)
            if pmi <= self.pmi_floor:
                continue
            w = float(pmi * cij)
            W[i, j] = w
            W[j, i] = w

        self._W = W.tocsr()
        self._build_transition()
        return self

    def _build_transition(self):
        if self._W is None:
            return
        row_sums = np.asarray(self._W.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sums)
        inv[row_sums > 0] = 1.0 / row_sums[row_sums > 0]
        self._T = diags(inv).dot(self._W).tocsr()

    def propagate(self, seed_ids: np.ndarray, seed_vals: np.ndarray | None = None,
                  hops: int = 2, decay: float = 0.5,
                  zero_seed: bool = True) -> np.ndarray:
        """Spreading activation over the graph.

        Args:
            seed_ids: indices of seed fragments
            seed_vals: activation values for each seed (default: all 1.0)
            hops: number of propagation steps
            decay: multiplicative decay per hop
            zero_seed: if True, zero out seed activations in the result (propagate mode)
                       if False, keep seed mass (diffuse mode)

        Returns:
            activation vector over all nodes
        """
        if self._T is None:
            raise RuntimeError("call finalize() before propagate()")
        n = self._T.shape[0]
        if seed_vals is None:
            seed_vals = np.ones(len(seed_ids), dtype=np.float32)
        act = np.zeros(n, dtype=np.float32)
        act[seed_ids] = seed_vals / (seed_vals.max() + 1e-12)

        total = act.copy()
        frontier = act.copy()
        for _ in range(hops):
            frontier = self._T.T.dot(frontier) * decay
            total = total + frontier

        if zero_seed:
            total[seed_ids] = 0.0
        return total

    def diffuse(self, seed_ids: np.ndarray, seed_vals: np.ndarray | None = None,
                hops: int = 2, decay: float = 0.5) -> np.ndarray:
        """Diffuse keeping seed mass (for uncertainty)."""
        return self.propagate(seed_ids, seed_vals, hops, decay, zero_seed=False)

    @property
    def has_edges(self) -> bool:
        if self._W is None:
            return False
        return self._W.nnz > 0

    @property
    def n_edges(self) -> int:
        if self._W is None:
            return 0
        return self._W.nnz // 2

    def edge_weight(self, i: int, j: int) -> float:
        if self._W is None:
            return 0.0
        return float(self._W[i, j])

    def state_dict(self) -> dict:
        return {
            "min_support": self.min_support,
            "pmi_floor": self.pmi_floor,
            "N": self.N,
            "ni": self.ni,
            "nij": {f"{k[0]},{k[1]}": v for k, v in self.nij.items()},
        }

    def load_state_dict(self, sd: dict):
        self.min_support = sd["min_support"]
        self.pmi_floor = sd["pmi_floor"]
        self.N = sd["N"]
        self.ni = sd["ni"]
        self.nij = {}
        for k_str, v in sd["nij"].items():
            a, b = k_str.split(",")
            self.nij[(int(a), int(b))] = v
        if self.N > 0:
            self.finalize()
