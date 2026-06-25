"""Context Window — working memory for the cognitive pipeline.

Stores recent events, world state snapshots, active entities, topic stack,
unresolved references, deltas, reasoning records, simulation outcomes, and
attention history — all with hard budget limits and recency decay.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    """Base fields for any item stored in ContextWindow collections."""
    importance: float = 0.5
    access_count: int = 0
    last_accessed_turn: int = 0


@dataclass
class ContextEvent(MemoryItem):
    modality: str = ""
    content: str = ""
    concepts: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    turn: int = 0


@dataclass
class EntityRecord(MemoryItem):
    name: str = ""
    confidence: float = 0.5
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    mention_count: int = 1


@dataclass
class TopicFrame:
    topic: str = ""
    confidence: float = 1.0
    turn_started: int = 0
    last_active_turn: int = 0


@dataclass
class Reference:
    surface: str = ""
    candidates: list[str] = field(default_factory=list)
    turn: int = 0
    resolved: bool = False


@dataclass
class WorldStateSnapshot(MemoryItem):
    concepts: dict[str, float] = field(default_factory=dict)
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    turn: int = 0
    timestamp: float = 0.0


@dataclass
class WorldDelta(MemoryItem):
    concept: str = ""
    attribute: str = ""
    old_value: Any = None
    new_value: Any = None
    turn: int = 0


@dataclass
class ReasoningRecord(MemoryItem):
    conclusion: str = ""
    concepts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    turn: int = 0


@dataclass
class SimRecord(MemoryItem):
    outcome: str = ""
    concepts: list[str] = field(default_factory=list)
    probability: float = 0.0
    turn: int = 0


@dataclass
class AttentionSnapshot(MemoryItem):
    attended: list[tuple[str, float]] = field(default_factory=list)
    suppressed: list[tuple[str, float]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    task_mode: str = "browsing"
    n_candidates: int = 0
    turn: int = 0


class ContextWindow:
    def __init__(
        self,
        max_events: int = 20,
        max_world_states: int = 10,
        max_entities: int = 30,
        max_topic_depth: int = 5,
        max_references: int = 10,
        max_deltas: int = 10,
        max_reasoning: int = 10,
        max_simulation: int = 5,
        max_attention_history: int = 20,
    ):
        self.max_events = max_events
        self.max_world_states = max_world_states
        self.max_entities = max_entities
        self.max_topic_depth = max_topic_depth
        self.max_references = max_references
        self.max_deltas = max_deltas
        self.max_reasoning = max_reasoning
        self.max_simulation = max_simulation
        self.max_attention_history = max_attention_history

        self._events: list[ContextEvent] = []
        self._world_states: list[WorldStateSnapshot] = []
        self.active_entities: dict[str, EntityRecord] = {}
        self.topic_stack: list[TopicFrame] = []
        self.unresolved_references: list[Reference] = []
        self._deltas: list[WorldDelta] = []
        self._reasoning: list[ReasoningRecord] = []
        self._simulations: list[SimRecord] = []
        self._attention_history: list[AttentionSnapshot] = []
        self.turn_counter: int = 0

    # ── Properties for external access (backward-compat) ─────────────

    @property
    def recent_events(self):
        return self._events

    @property
    def recent_world_states(self):
        return self._world_states

    @property
    def recent_world_deltas(self):
        return self._deltas

    @property
    def recent_reasoning(self):
        return self._reasoning

    @property
    def recent_simulation(self):
        return self._simulations

    @property
    def attention_history(self):
        return self._attention_history

    # ── Memory score ─────────────────────────────────────────────────

    def _compute_memory_score(self, item) -> float:
        if self.turn_counter == 0:
            return item.importance
        recency = 1.0 / (1.0 + self.turn_counter - item.last_accessed_turn)
        return item.importance * 0.5 + recency * 0.2 + item.access_count * 0.3

    def _evict_lowest_score(self, items: list, max_size: int) -> None:
        while len(items) > max_size:
            lowest = min(items, key=lambda x: self._compute_memory_score(x))
            items.remove(lowest)

    # ── Event ingestion ──────────────────────────────────────────────

    def ingest_event(self, modality: str, content: str, concepts: list[str], importance: float = 0.5) -> None:
        event = ContextEvent(
            modality=modality,
            content=content,
            concepts=concepts,
            importance=importance,
            timestamp=time.time(),
            turn=self.turn_counter,
            last_accessed_turn=self.turn_counter,
        )
        self._events.append(event)
        self._evict_lowest_score(self._events, self.max_events)
        self.update_entities(concepts)

    def update_entities(self, concept_names: list[str]) -> None:
        now = self.turn_counter
        for name in set(concept_names):
            if not name:
                continue
            if name in self.active_entities:
                rec = self.active_entities[name]
                rec.last_seen_turn = now
                rec.mention_count += 1
                rec.access_count += 1
            else:
                self.active_entities[name] = EntityRecord(
                    name=name,
                    importance=0.5,
                    first_seen_turn=now,
                    last_seen_turn=now,
                    last_accessed_turn=now,
                )
        # Evict lowest score entities if over budget
        if len(self.active_entities) > self.max_entities:
            sorted_entities = sorted(
                self.active_entities.values(),
                key=lambda e: self._compute_memory_score(e),
            )
            n_evict = len(self.active_entities) - self.max_entities
            for rec in sorted_entities[:n_evict]:
                del self.active_entities[rec.name]

    # ── World state ──────────────────────────────────────────────────

    def store_world_state(
        self,
        concepts: dict[str, float],
        relations: list[tuple[str, str, str]],
        importance: float = 0.6,
    ) -> None:
        snapshot = WorldStateSnapshot(
            concepts=concepts,
            relations=relations,
            importance=importance,
            turn=self.turn_counter,
            timestamp=time.time(),
            last_accessed_turn=self.turn_counter,
        )
        self._world_states.append(snapshot)
        self._evict_lowest_score(self._world_states, self.max_world_states)

    def add_world_delta(self, concept: str, attribute: str, old: Any, new: Any, importance: float = 0.4) -> None:
        self._deltas.append(
            WorldDelta(concept=concept, attribute=attribute,
                       old_value=old, new_value=new, turn=self.turn_counter,
                       importance=importance, last_accessed_turn=self.turn_counter)
        )
        self._evict_lowest_score(self._deltas, self.max_deltas)

    # ── Topics ────────────────────────────────────────────────────────

    def push_topic(self, topic: str, confidence: float = 1.0) -> None:
        now = self.turn_counter
        if self.topic_stack and self.topic_stack[-1].topic == topic:
            self.topic_stack[-1].last_active_turn = now
            self.topic_stack[-1].confidence = max(self.topic_stack[-1].confidence, confidence)
            return
        self.topic_stack.append(
            TopicFrame(topic=topic, confidence=confidence, turn_started=now, last_active_turn=now)
        )
        if len(self.topic_stack) > self.max_topic_depth:
            self.topic_stack.pop(0)

    def pop_topic(self) -> str | None:
        if self.topic_stack:
            return self.topic_stack.pop().topic
        return None

    # ── References ────────────────────────────────────────────────────

    def add_reference(self, surface: str, candidates: list[str]) -> None:
        self.unresolved_references.append(
            Reference(surface=surface, candidates=candidates, turn=self.turn_counter)
        )
        if len(self.unresolved_references) > self.max_references:
            # Evict oldest unresolved
            unresolved = [r for r in self.unresolved_references if not r.resolved]
            if unresolved:
                self.unresolved_references.remove(unresolved[0])

    def resolve_reference(self, surface: str, resolved_name: str) -> None:
        for ref in self.unresolved_references:
            if ref.surface == surface and not ref.resolved:
                ref.resolved = True
                break

    # ── Reasoning / Simulation / Attention ────────────────────────────

    def add_reasoning(self, conclusion: str, concepts: list[str], confidence: float, importance: float = 0.7) -> None:
        self._reasoning.append(
            ReasoningRecord(conclusion=conclusion, concepts=concepts,
                           confidence=confidence, turn=self.turn_counter,
                           importance=importance, last_accessed_turn=self.turn_counter)
        )
        self._evict_lowest_score(self._reasoning, self.max_reasoning)

    def add_simulation(self, outcome: str, concepts: list[str], probability: float, importance: float = 0.5) -> None:
        self._simulations.append(
            SimRecord(outcome=outcome, concepts=concepts,
                     probability=probability, turn=self.turn_counter,
                     importance=importance, last_accessed_turn=self.turn_counter)
        )
        self._evict_lowest_score(self._simulations, self.max_simulation)

    def add_attention_snapshot(
        self,
        attended: list[tuple[str, float]],
        suppressed: list[tuple[str, float]],
        weights: dict[str, float] | None = None,
        task_mode: str = "browsing",
        n_candidates: int = 0,
        importance: float = 0.3,
    ) -> None:
        self._attention_history.append(
            AttentionSnapshot(
                attended=attended,
                suppressed=suppressed,
                weights=weights or {},
                task_mode=task_mode,
                n_candidates=n_candidates,
                turn=self.turn_counter,
                importance=importance,
                last_accessed_turn=self.turn_counter,
            )
        )
        self._evict_lowest_score(self._attention_history, self.max_attention_history)

    # ── Summary / export ──────────────────────────────────────────────

    def get_context_summary(self) -> dict[str, Any]:
        entities = [
            {"name": e.name, "confidence": e.confidence,
             "mention_count": e.mention_count, "last_seen": e.last_seen_turn}
            for e in sorted(self.active_entities.values(),
                          key=lambda x: x.mention_count, reverse=True)
        ]
        return {
            "turn": self.turn_counter,
            "n_events": len(self.recent_events),
            "topic_stack": [t.topic for t in self.topic_stack],
            "entities": entities,
            "n_world_states": len(self.recent_world_states),
            "unresolved_refs": [
                {"surface": r.surface, "candidates": r.candidates}
                for r in self.unresolved_references if not r.resolved
            ],
            "n_reasoning": len(self.recent_reasoning),
            "n_simulation": len(self.recent_simulation),
        }

    def format_context_block(self) -> str:
        lines = [f"=== Context (turn {self.turn_counter}) ==="]
        if self.topic_stack:
            topics = " > ".join(t.topic for t in self.topic_stack)
            lines.append(f"Topics: {topics}")
        if self.active_entities:
            top_entities = sorted(self.active_entities.values(),
                                key=lambda e: e.mention_count, reverse=True)[:5]
            lines.append("Entities: " + ", ".join(f"{e.name}({e.mention_count})" for e in top_entities))
        if self.recent_reasoning:
            lines.append("Recent reasoning:")
            for r in list(self.recent_reasoning)[-3:]:
                lines.append(f"  - {r.conclusion}")
        if self.recent_simulation:
            lines.append("Recent simulations:")
            for s in list(self.recent_simulation)[-2:]:
                lines.append(f"  - {s.outcome} (p={s.probability:.2f})")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "turn_counter": self.turn_counter,
            "events": [
                {"modality": e.modality, "content": e.content,
                 "concepts": e.concepts, "timestamp": e.timestamp, "turn": e.turn,
                 "importance": e.importance, "access_count": e.access_count,
                 "last_accessed_turn": e.last_accessed_turn}
                for e in self._events
            ],
            "world_states": [
                {"concepts": s.concepts, "relations": s.relations,
                 "turn": s.turn, "timestamp": s.timestamp,
                 "importance": s.importance, "access_count": s.access_count,
                 "last_accessed_turn": s.last_accessed_turn}
                for s in self._world_states
            ],
            "entities": {
                name: {"name": e.name, "confidence": e.confidence,
                       "first_seen_turn": e.first_seen_turn,
                       "last_seen_turn": e.last_seen_turn,
                       "mention_count": e.mention_count,
                       "importance": e.importance, "access_count": e.access_count,
                       "last_accessed_turn": e.last_accessed_turn}
                for name, e in self.active_entities.items()
            },
            "topics": [
                {"topic": t.topic, "confidence": t.confidence,
                 "turn_started": t.turn_started, "last_active_turn": t.last_active_turn}
                for t in self.topic_stack
            ],
            "references": [
                {"surface": r.surface, "candidates": r.candidates,
                 "turn": r.turn, "resolved": r.resolved}
                for r in self.unresolved_references
            ],
            "deltas": [
                {"concept": d.concept, "attribute": d.attribute,
                 "old_value": d.old_value, "new_value": d.new_value, "turn": d.turn,
                 "importance": d.importance, "access_count": d.access_count,
                 "last_accessed_turn": d.last_accessed_turn}
                for d in self._deltas
            ],
            "reasoning": [
                {"conclusion": r.conclusion, "concepts": r.concepts,
                 "confidence": r.confidence, "turn": r.turn,
                 "importance": r.importance, "access_count": r.access_count,
                 "last_accessed_turn": r.last_accessed_turn}
                for r in self._reasoning
            ],
            "simulations": [
                {"outcome": s.outcome, "concepts": s.concepts,
                 "probability": s.probability, "turn": s.turn,
                 "importance": s.importance, "access_count": s.access_count,
                 "last_accessed_turn": s.last_accessed_turn}
                for s in self._simulations
            ],
            "attention_history": [
                {"attended": a.attended, "suppressed": a.suppressed,
                 "weights": a.weights, "task_mode": a.task_mode,
                 "n_candidates": a.n_candidates, "turn": a.turn,
                 "importance": a.importance, "access_count": a.access_count,
                 "last_accessed_turn": a.last_accessed_turn}
                for a in self._attention_history
            ],
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.turn_counter = sd.get("turn_counter", 0)
        self._events.clear()
        for e in sd.get("events", []):
            self._events.append(ContextEvent(**e))
        self._world_states.clear()
        for s in sd.get("world_states", []):
            self._world_states.append(WorldStateSnapshot(**s))
        self.active_entities.clear()
        for name, e in sd.get("entities", {}).items():
            self.active_entities[name] = EntityRecord(**e)
        self.topic_stack.clear()
        for t in sd.get("topics", []):
            self.topic_stack.append(TopicFrame(**t))
        self.unresolved_references.clear()
        for r in sd.get("references", []):
            self.unresolved_references.append(Reference(**r))
        self._deltas.clear()
        for d in sd.get("deltas", []):
            self._deltas.append(WorldDelta(**d))
        self._reasoning.clear()
        for r in sd.get("reasoning", []):
            self._reasoning.append(ReasoningRecord(**r))
        self._simulations.clear()
        for s in sd.get("simulations", []):
            self._simulations.append(SimRecord(**s))
        self._attention_history.clear()
        for a in sd.get("attention_history", []):
            self._attention_history.append(AttentionSnapshot(**a))

    def reset(self) -> None:
        self._events.clear()
        self._world_states.clear()
        self.active_entities.clear()
        self.topic_stack.clear()
        self.unresolved_references.clear()
        self._deltas.clear()
        self._reasoning.clear()
        self._simulations.clear()
        self._attention_history.clear()
        self.turn_counter = 0

    def increment_turn(self) -> None:
        self.turn_counter += 1
