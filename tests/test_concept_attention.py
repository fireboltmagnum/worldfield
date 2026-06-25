"""Tests for hierarchical ConceptAttention."""
from __future__ import annotations

import pytest
import numpy as np

from worldfield.core.concept_attention import (
    ConceptAttention, AttentionResult, ScoredConcept,
)
from worldfield.core.world_graph import WorldGraph


@pytest.fixture
def graph():
    g = WorldGraph()
    # Add some concepts
    g.add_concept("cat", canonical_name="cat")
    g.add_concept("mat", canonical_name="mat")
    g.add_concept("dog", canonical_name="dog")
    g.add_concept("bone", canonical_name="bone")
    g.add_concept("sofa", canonical_name="sofa")
    # Add edges
    g.add_relation("cat", "sits_on", "mat", confidence=0.9)
    g.add_relation("dog", "chews", "bone", confidence=0.8)
    g.add_relation("cat", "sleeps_on", "sofa", confidence=0.7)
    return g


@pytest.fixture
def context_window():
    from worldfield.core.context_window import ContextWindow
    cw = ContextWindow(max_events=5, max_world_states=3, max_entities=10)
    cw.ingest_event("text", "the cat sat on the mat", ["cat", "mat"])
    cw.store_world_state({"cat": 0.9, "mat": 0.8}, [("cat", "sits_on", "mat")])
    cw.push_topic("pets", 0.9)
    return cw


def make_encoder():
    """Simple embedding: one-hot-like based on character hash."""
    dim = 16
    def encode(text: str) -> np.ndarray:
        vec = np.zeros(dim)
        for i, ch in enumerate(text):
            vec[hash(ch) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    return encode


class FakeGoal:
    def __init__(self, description, embedding=None):
        self.description = description
        self.embedding = embedding
        self.status = "active"
        self.priority = 1


class FakeGoalManager:
    def __init__(self, goals=None):
        self._goals = goals or []
    def get_active_goals(self):
        return self._goals


def test_attention_returns_attention_result(graph, context_window):
    encoder = make_encoder()
    goal_mgr = FakeGoalManager()
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=3)
    result = attn.hierarchical_attend(
        active_goals=goal_mgr.get_active_goals(),
        context_window=context_window,
        current_concepts=["cat", "dog"],
    )
    assert isinstance(result, AttentionResult)
    assert len(result.attended) <= 3
    assert result.n_candidates > 0


def test_attention_no_goals_fallback_to_context(graph, context_window):
    encoder = make_encoder()
    goal_mgr = FakeGoalManager()
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=3)
    result = attn.hierarchical_attend(
        active_goals=[],
        context_window=context_window,
        current_concepts=["unknown"],
    )
    # Should fall back to context entities: cat, mat
    assert result.n_candidates > 0
    assert any(s.name in ("cat", "mat") for s in result.attended)


def test_attention_goal_filter(graph, context_window):
    encoder = make_encoder()
    enc = encoder("dog bone")
    goal_mgr = FakeGoalManager([FakeGoal("dog bone", embedding=enc)])
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=3)
    result = attn.hierarchical_attend(
        active_goals=goal_mgr.get_active_goals(),
        context_window=context_window,
        current_concepts=[],
    )
    assert result.n_candidates > 0
    attended_names = [s.name for s in result.attended]
    assert "dog" in attended_names or "bone" in attended_names


def test_attention_top_k_respected(graph, context_window):
    encoder = make_encoder()
    goal_mgr = FakeGoalManager()
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=2)
    result = attn.hierarchical_attend(
        active_goals=[],
        context_window=context_window,
        current_concepts=["cat", "dog", "mat", "bone"],
    )
    assert len(result.attended) == 2


def test_attention_suppressed_has_remaining(graph, context_window):
    encoder = make_encoder()
    goal_mgr = FakeGoalManager()
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=1)
    result = attn.hierarchical_attend(
        active_goals=[],
        context_window=context_window,
        current_concepts=["cat", "dog", "mat"],
    )
    assert len(result.suppressed) > 0


def test_attention_task_aware_weighting(graph, context_window):
    encoder = make_encoder()
    enc = encoder("build a house")
    goal_mgr = FakeGoalManager([FakeGoal("build a house", embedding=enc)])
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=3)
    result = attn.hierarchical_attend(
        active_goals=goal_mgr.get_active_goals(),
        context_window=context_window,
        current_concepts=["cat", "dog"],
    )
    assert "goal_alignment" in result.weights
    assert result.task_mode != "browsing"


def test_attention_graceful_degradation(graph, context_window):
    """When goal filter produces zero candidates, falls back to context."""
    encoder = make_encoder()
    # Embedding with no relation to any graph node
    weird_goal_enc = np.zeros(16)
    goal_mgr = FakeGoalManager([FakeGoal("xyznonexistent", embedding=weird_goal_enc)])
    attn = ConceptAttention(graph=graph, encoder=encoder, top_k=3)
    result = attn.hierarchical_attend(
        active_goals=goal_mgr.get_active_goals(),
        context_window=context_window,
        current_concepts=[],
    )
    assert result.n_candidates > 0  # fell back to context
