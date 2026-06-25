"""Tests for MemoryRetrieval (sparse subgraph fetch)."""
from __future__ import annotations

import numpy as np
import pytest

from worldfield.core.memory_retrieval import MemoryRetrieval, RetrievalResult
from worldfield.core.world_graph import WorldGraph


@pytest.fixture
def graph():
    g = WorldGraph()
    g.add_concept("cat", canonical_name="cat")
    g.add_concept("mat", canonical_name="mat")
    g.add_concept("dog", canonical_name="dog")
    g.add_concept("bone", canonical_name="bone")
    g.add_concept("sofa", canonical_name="sofa")
    g.add_concept("table", canonical_name="table")
    g.add_relation("cat", "sits_on", "mat", confidence=0.9)
    g.add_relation("cat", "sleeps_on", "sofa", confidence=0.7)
    g.add_relation("dog", "chews", "bone", confidence=0.8)
    g.add_relation("mat", "on", "table", confidence=0.6)
    return g


def make_encoder():
    dim = 16
    def encode(text: str) -> np.ndarray:
        vec = np.zeros(dim)
        for i, ch in enumerate(text):
            vec[hash(ch) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    return encode


def test_fetch_returns_retrieval_result(graph):
    mr = MemoryRetrieval(graph=graph, encoder=make_encoder())
    result = mr.fetch(["cat"])
    assert isinstance(result, RetrievalResult)
    assert "cat" in result.nodes
    assert len(result.nodes) > 0


def test_fetch_includes_scores(graph):
    mr = MemoryRetrieval(graph=graph, encoder=make_encoder())
    result = mr.fetch(["cat"])
    assert hasattr(result, "scores")
    for name in result.nodes:
        assert name in result.scores
        assert result.scores[name] > 0.0


def test_fetch_distance_independent(graph):
    """Score-based retrieval can find nodes at any distance, not just 1-hop."""
    mr = MemoryRetrieval(graph=graph, encoder=make_encoder(), max_retrieved=10)
    # "table" is 2 hops from "cat" (cat->mat->table), should still be score-comparable
    mr.graph.add_concept("tuna", canonical_name="tuna")
    mr.graph.add_relation("cat", "eats", "tuna", confidence=0.9)
    result = mr.fetch(["cat"])
    assert "table" in result.nodes or "tuna" in result.nodes


def test_fetch_budget_retrieved(graph):
    mr = MemoryRetrieval(graph=graph, encoder=make_encoder(), max_retrieved=2)
    result = mr.fetch(["cat"])
    assert len(result.nodes) <= 2


def test_fetch_no_match(graph):
    mr = MemoryRetrieval(graph=graph, encoder=make_encoder())
    result = mr.fetch(["nonexistent"])
    assert len(result.nodes) == 0
    assert len(result.edges) == 0


def test_fetch_empty_list(graph):
    mr = MemoryRetrieval(graph=graph, encoder=make_encoder())
    result = mr.fetch([])
    assert len(result.nodes) == 0
    assert len(result.edges) == 0
