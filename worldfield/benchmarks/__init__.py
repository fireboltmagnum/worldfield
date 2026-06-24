"""Benchmarks runner for WorldField."""
from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..core.engine import Engine


def run_benchmarks(engine: Engine, cfg: Config):
    print("=" * 50)
    print("WorldField Benchmarks")
    print("=" * 50)

    # 1. Encoding speed
    print("\n--- Encoding Speed ---")
    texts = [f"test query {i}" for i in range(10)]
    start = time.perf_counter()
    for t in texts:
        engine.text_encoder.encode(t)
    elapsed = time.perf_counter() - start
    print(f"  Text: {len(texts)} in {elapsed:.3f}s ({elapsed/len(texts)*1000:.1f}ms each)")

    # 2. Fragment store retrieval
    print("\n--- Fragment Retrieval ---")
    n_test = min(engine.store.count, 10)
    if n_test > 0:
        vec = engine.text_encoder.encode("test")
        start = time.perf_counter()
        for _ in range(10):
            engine.store.search(vec, k=5)
        elapsed = time.perf_counter() - start
        print(f"  10 searches in {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")
    else:
        print("  (no fragments stored yet)")

    # 3. Graph propagation
    print("\n--- Graph Propagation ---")
    if engine.graph.has_edges:
        scores = engine.graph.propagate(np.array([0]), hops=cfg.refine_hops)
        print(f"  Propagation over {engine.graph.n_edges} edges: OK")
        print(f"  Top score: {float(np.max(scores)):.4f}")
    else:
        print("  (no graph edges yet)")

    # 4. Full pipeline latency
    print("\n--- Full Pipeline Latency ---")
    if engine.store.count > 0:
        start = time.perf_counter()
        result = engine.process("benchmark test")
        elapsed = time.perf_counter() - start
        print(f"  Full process: {elapsed*1000:.1f}ms")
        print(f"  Slots active: {result['slot_state_active']}")
        print(f"  Related: {len(result['related'])}")
    else:
        print("  (storing test fragment...)")
        start = time.perf_counter()
        result = engine.process("benchmark test")
        elapsed = time.perf_counter() - start
        print(f"  Full process: {elapsed*1000:.1f}ms")

    print("\n" + "=" * 50)
    print("Benchmarks complete.")
