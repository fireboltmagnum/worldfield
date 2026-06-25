"""Tests for ContextHorizon consolidation layer."""
from __future__ import annotations

import pytest
from worldfield.core.context_horizon import ContextHorizon
from worldfield.core.concept_attention import ScoredConcept
from worldfield.core.memory_retrieval import RetrievalResult
from worldfield.core.context_window import ReasoningRecord


def test_horizon_created_empty():
    h = ContextHorizon()
    assert h.current_input == []
    assert h.attended_concepts == []
    assert h.get_all_concepts() == set()


def test_horizon_get_all_concepts():
    h = ContextHorizon(
        current_input=["cat", "dog"],
        attended_concepts=[ScoredConcept(name="mat", score=0.9)],
        world_state={"sofa": 0.8},
    )
    all_c = h.get_all_concepts()
    assert "cat" in all_c
    assert "dog" in all_c
    assert "mat" in all_c
    assert "sofa" in all_c


def test_horizon_retrieved_memory_contributes():
    mr = RetrievalResult()
    mr.nodes = {"bone": None, "ball": None}
    h = ContextHorizon(
        current_input=["dog"],
        retrieved_memory=mr,
    )
    all_c = h.get_all_concepts()
    assert "bone" in all_c
    assert "ball" in all_c


def test_horizon_to_dict_includes_all_slots():
    h = ContextHorizon(
        current_input=["cat"],
        attended_concepts=[ScoredConcept(name="mat", score=0.9)],
        world_state={"sofa": 0.8},
    )
    d = h.to_dict()
    assert "current_input" in d
    assert "attended_concepts" in d
    assert "world_state" in d


def test_horizon_format_block():
    h = ContextHorizon(
        current_input=["cat"],
        world_state={"cat": 0.9, "mat": 0.8},
    )
    block = h.format_horizon_block()
    assert "cat" in block
    assert "mat" in block
