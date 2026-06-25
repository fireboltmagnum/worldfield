"""Tests for ContextWindow (working memory)."""
from __future__ import annotations

import pytest
import numpy as np

from worldfield.core.context_window import (
    ContextWindow, ContextEvent, EntityRecord, TopicFrame,
    Reference, WorldStateSnapshot, WorldDelta,
)


def test_ingest_event_stores_and_tracks_entities():
    cw = ContextWindow()
    cw.ingest_event("text", "hello world", ["cat", "mat"])
    assert len(cw.recent_events) == 1
    assert cw.recent_events[0].content == "hello world"
    assert cw.recent_events[0].concepts == ["cat", "mat"]
    assert "cat" in cw.active_entities
    assert cw.active_entities["cat"].mention_count == 1


def test_ingest_multiple_events_budget():
    cw = ContextWindow(max_events=3)
    for i in range(5):
        cw.ingest_event("text", f"msg {i}", [f"concept_{i}"])
    assert len(cw.recent_events) == 3
    # Lowest-score event should be evicted (recent high-importance survive)

def test_importance_retention():
    """High-importance items survive eviction over low-importance items."""
    cw = ContextWindow(max_events=3)
    cw.ingest_event("text", "important msg", ["key"], importance=0.9)
    cw.ingest_event("text", "low importance", ["trash"], importance=0.1)
    # Artificially inflate the low-importance event's age
    cw.recent_events[-1].access_count = 0
    cw.recent_events[-1].last_accessed_turn = -10
    # Add enough new events to trigger eviction
    for i in range(3):
        cw.ingest_event("text", f"new msg {i}", [f"new_{i}"], importance=0.5)
    # Low-importance "trash" should be evicted, high-importance "key" stays
    surviving = [e.content for e in cw.recent_events]
    assert "important msg" in surviving, "High-importance item evicted!"
    assert "low importance" not in surviving, "Low-importance item survived!"

def test_memory_score_computation():
    cw = ContextWindow()
    cw.ingest_event("text", "test", ["cat"])
    event = cw.recent_events[0]
    score = cw._compute_memory_score(event)
    assert 0.0 <= score <= 1.0


def test_entity_mention_count_tracking():
    cw = ContextWindow()
    cw.ingest_event("text", "first", ["cat"])
    cw.ingest_event("text", "second", ["cat", "dog"])
    assert cw.active_entities["cat"].mention_count == 2
    assert cw.active_entities["dog"].mention_count == 1
    assert cw.active_entities["cat"].last_seen_turn == 0


def test_entity_budget_evicts_least_recently_seen():
    cw = ContextWindow(max_entities=3)
    for i, name in enumerate(["a", "b", "c", "d"]):
        cw.ingest_event("text", f"msg {i}", [name])
    assert len(cw.active_entities) == 3
    assert "a" not in cw.active_entities  # evicted (oldest last_seen)


def test_topic_stack_push_and_depth():
    cw = ContextWindow(max_topic_depth=3)
    cw.push_topic("cats", 0.9)
    cw.push_topic("dogs", 0.8)
    cw.push_topic("birds", 0.7)
    cw.push_topic("fish", 0.6)  # should evict oldest
    assert len(cw.topic_stack) == 3
    assert cw.topic_stack[0].topic == "dogs"  # "cats" popped


def test_reference_add_and_resolve():
    cw = ContextWindow()
    cw.add_reference("it", ["cat", "dog"])
    assert len(cw.unresolved_references) == 1
    assert cw.unresolved_references[0].surface == "it"
    assert not cw.unresolved_references[0].resolved
    cw.resolve_reference("it", "cat")
    assert cw.unresolved_references[0].resolved


def test_store_world_state():
    cw = ContextWindow(max_world_states=3)
    cw.store_world_state({"cat": 0.9, "mat": 0.8}, [("cat", "sits_on", "mat")])
    assert len(cw.recent_world_states) == 1
    snap = cw.recent_world_states[0]
    assert snap.concepts == {"cat": 0.9, "mat": 0.8}
    assert snap.relations == [("cat", "sits_on", "mat")]


def test_world_state_budget():
    cw = ContextWindow(max_world_states=2)
    for i in range(4):
        cw.store_world_state({f"c{i}": 1.0}, [])
    assert len(cw.recent_world_states) == 2


def test_add_reasoning_and_simulation():
    cw = ContextWindow(max_reasoning=2, max_simulation=2)
    cw.add_reasoning("conclusion", ["cat"], 0.9)
    cw.add_reasoning("another", ["dog"], 0.8)
    cw.add_reasoning("third", ["bird"], 0.7)  # evicts oldest
    assert len(cw.recent_reasoning) == 2
    assert cw.recent_reasoning[0].conclusion == "another"
    cw.add_simulation("outcome", ["cat"], 0.6)
    assert len(cw.recent_simulation) == 1


def test_add_attention_snapshot():
    cw = ContextWindow()
    cw.add_attention_snapshot(
        attended=[("cat", 0.9)],
        suppressed=[("dog", 0.3)],
        task_mode="browsing",
    )
    assert len(cw.attention_history) == 1
    snap = cw.attention_history[0]
    assert snap.attended == [("cat", 0.9)]
    assert snap.task_mode == "browsing"


def test_get_context_summary():
    cw = ContextWindow()
    cw.ingest_event("text", "hello", ["cat"])
    cw.push_topic("pets", 0.9)
    summary = cw.get_context_summary()
    assert "cat" in [e["name"] for e in summary["entities"]]
    assert summary["topic_stack"] == ["pets"]
    assert summary["turn"] == 0
    assert summary["n_events"] == 1


def test_state_dict_roundtrip():
    cw = ContextWindow()
    cw.ingest_event("text", "hello", ["cat"])
    cw.push_topic("pets", 0.9)
    cw.store_world_state({"cat": 0.8}, [])
    sd = cw.state_dict()
    cw2 = ContextWindow()
    cw2.load_state_dict(sd)
    assert len(cw2.recent_events) == 1
    assert cw2.recent_events[0].content == "hello"
    assert len(cw2.recent_world_states) == 1
    assert cw2.topic_stack[0].topic == "pets"
    assert "cat" in cw2.active_entities
    assert cw2.turn_counter == 0  # same turn


def test_reset():
    cw = ContextWindow()
    cw.ingest_event("text", "hello", ["cat"])
    cw.reset()
    assert len(cw.recent_events) == 0
    assert len(cw.active_entities) == 0


def test_world_delta():
    cw = ContextWindow(max_deltas=2)
    cw.add_world_delta("cat", "confidence", 0.9, 0.5)
    cw.add_world_delta("dog", "confidence", 0.8, 0.6)
    cw.add_world_delta("bird", "confidence", 0.7, 0.4)
    assert len(cw.recent_world_deltas) == 2
    assert cw.recent_world_deltas[0].concept == "dog"
