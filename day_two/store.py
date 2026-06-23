"""The dumbest fragment store that could possibly work (plan §3, §8).

A fragment = latent vector + metadata. The store wraps a FAISS index for
approximate retrieval and keeps the metadata in parallel arrays. No learning,
no decay logic yet — just store, retrieve, and an activation rule that decides
how many fragments "wake up". Sparsity is a property of THAT rule, not of FAISS.
"""
import time
import numpy as np
import faiss


class FragmentStore:
    def __init__(self, dim, use_hnsw=False):
        self.dim = dim
        # cosine similarity via inner product on L2-normalized vectors
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
        else:
            self.index = faiss.IndexFlatIP(dim)  # exact; baseline for honesty
        # parallel metadata (plan §3)
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int64)   # ground-truth class (for eval)
        self.strength = np.empty((0,), dtype=np.float32)
        self.last_used = np.empty((0,), dtype=np.int64)
        self._clock = 0

    @staticmethod
    def _normalize(x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        faiss.normalize_L2(x)
        return x

    def add(self, vectors, labels):
        v = self._normalize(vectors)
        self.index.add(v)
        n = v.shape[0]
        self.vectors = np.vstack([self.vectors, v])
        self.labels = np.concatenate([self.labels, np.asarray(labels, dtype=np.int64)])
        self.strength = np.concatenate([self.strength, np.ones(n, dtype=np.float32)])
        self.last_used = np.concatenate([self.last_used, np.zeros(n, dtype=np.int64)])

    def __len__(self):
        return self.index.ntotal

    def search(self, queries, k):
        """Return (similarities, indices, latency_ms_per_query)."""
        q = self._normalize(queries.copy())
        t0 = time.perf_counter()
        sims, idx = self.index.search(q, k)
        dt = (time.perf_counter() - t0) * 1000 / q.shape[0]
        return sims, idx, dt

    def activate(self, queries, k_max, sim_threshold):
        """The activation rule (plan §9): retrieve up to k_max neighbors, then
        keep only those above a similarity threshold. THIS is where sparsity
        comes from — not from FAISS. Returns a list of active index arrays and
        bumps usage metadata for activated fragments.
        """
        sims, idx, _ = self.search(queries, k_max)
        active = []
        for row_sims, row_idx in zip(sims, idx):
            keep = row_idx[(row_sims >= sim_threshold) & (row_idx >= 0)]
            active.append(keep)
            self._clock += 1
            if keep.size:
                self.strength[keep] += 1.0
                self.last_used[keep] = self._clock
        return active
