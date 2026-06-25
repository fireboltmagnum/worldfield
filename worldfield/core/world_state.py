"""World State — the system's current reality model.

Built from activated concepts + current input + graph knowledge, the world state
represents what the system believes *right now*. It is ephemeral, hypothesis-based,
and modality-independent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .world_graph import WorldGraph


@dataclass
class RelationBelief:
    """A believed relation between two entities at the current moment."""
    source: str
    predicate: str
    target: str
    confidence: float = 0.5
    source_modality: str = "text"
    evidence_ids: list[str] = field(default_factory=list)
    polarity: bool = True


class WorldState:
    """A snapshot of what the system believes at this instant.

    Parameters
    ----------
    entities:
        ``{entity_name: confidence}``
    relations:
        Believed relations currently in focus.
    attributes:
        ``{entity_name: {attribute_name: confidence}}``
    alternative_hypotheses:
        Competing world states when multiple interpretations exist.
    timestamp:
        Seconds since epoch.
    """

    def __init__(
        self,
        entities: dict[str, float] | None = None,
        relations: list[RelationBelief] | None = None,
        attributes: dict[str, dict[str, float]] | None = None,
        alternative_hypotheses: list[WorldState] | None = None,
        timestamp: float | None = None,
    ):
        self.entities: dict[str, float] = entities or {}
        self.relations: list[RelationBelief] = relations or []
        self.attributes: dict[str, dict[str, float]] = attributes or {}
        self.alternative_hypotheses: list[WorldState] = alternative_hypotheses or []
        self.timestamp: float = timestamp or time.time()

    # ── Queries ───────────────────────────────────────────────────────

    def get_entity(self, name: str) -> float | None:
        return self.entities.get(name)

    def has_entity(self, name: str) -> bool:
        return name in self.entities

    def get_relations(
        self,
        source: str | None = None,
        predicate: str | None = None,
        target: str | None = None,
    ) -> list[RelationBelief]:
        """Filter relations by any combination of fields."""
        result = self.relations
        if source:
            result = [r for r in result if r.source == source]
        if predicate:
            result = [r for r in result if r.predicate == predicate]
        if target:
            result = [r for r in result if r.target == target]
        return result

    def get_belief(
        self, source: str, predicate: str, target: str
    ) -> RelationBelief | None:
        """Return the belief for an exact triple, or *None*."""
        for r in self.relations:
            if r.source == source and r.predicate == predicate and r.target == target:
                return r
        return None

    def has_conflict(self) -> bool:
        """*True* when multiple relations connect the same source-target pair."""
        seen: set[tuple[str, str]] = set()
        for r in self.relations:
            key = (r.source, r.target)
            if key in seen:
                return True
            seen.add(key)
        return len(self.alternative_hypotheses) > 0

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": dict(self.entities),
            "relations": [
                {
                    "source": r.source,
                    "predicate": r.predicate,
                    "target": r.target,
                    "confidence": r.confidence,
                    "modality": r.source_modality,
                    "polarity": r.polarity,
                    "evidence": r.evidence_ids,
                }
                for r in self.relations
            ],
            "attributes": {e: dict(attrs) for e, attrs in self.attributes.items()},
            "n_alternatives": len(self.alternative_hypotheses),
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"WorldState(entities={len(self.entities)}, "
            f"relations={len(self.relations)}, "
            f"alternatives={len(self.alternative_hypotheses)})"
        )


class WorldStateBuilder:
    """Constructs :class:`WorldState` snapshots from activations and graph knowledge.

    Typical usage::

        builder = WorldStateBuilder(graph)
        state = builder.from_activations(activated, current_relations)
    """

    _PROPERTY_PREDICATES = frozenset({
        "is", "has", "is_a", "has_attribute", "has_property",
        "is_of", "attribute_of",
    })

    def __init__(self, graph: WorldGraph):
        self.graph = graph
        self._history: list[WorldState] = []

    def from_activations(
        self,
        activated: list[tuple[str, float]],
        current_relations: list[dict[str, Any]],
    ) -> WorldState:
        """Build a world state from the current activation snapshot.

        Parameters
        ----------
        activated:
            ``[(concept_name, activation_level), ...]`` from the activation engine.
        current_relations:
            Raw relation dicts from the current input resolution.

        Returns
        -------
        WorldState
            The system's current belief snapshot.
        """
        # 1. Collect known entity confidences
        entities: dict[str, float] = {}
        for name, level in activated:
            node = self.graph.get_concept(name)
            if node is not None and node.confidence > 0:
                entities[name] = node.confidence
            else:
                entities[name] = min(level, 1.0)

        for r in current_relations:
            for name in (r["source"], r["target"]):
                if name not in entities:
                    node = self.graph.get_concept(name)
                    entities[name] = node.confidence if node else 0.1

        # 2. Collect graph relations for activated concepts.
        #    Walk adjacency directly to distinguish forward vs reverse edges.
        graph_relation_dict: dict[tuple[str, str, str], RelationBelief] = {}
        for name, _ in activated:
            node = self.graph.get_concept(name)
            if node is None or node.id not in self.graph.adjacency:
                continue
            nid = node.id
            for pred, eids in self.graph.adjacency[nid].items():
                is_reverse = pred.startswith("~")
                actual_pred = pred[1:] if is_reverse else pred
                for eid in eids:
                    edge = self.graph.edges[eid]
                    if is_reverse:
                        # Edge points toward nid; other end is the source
                        src = self.graph.nodes.get(edge.source_id)
                        if src is None:
                            continue
                        src_name = src.canonical_name
                        tgt_name = name
                    else:
                        # Edge points away from nid
                        tgt = self.graph.nodes.get(edge.target_id)
                        if tgt is None:
                            continue
                        src_name = name
                        tgt_name = tgt.canonical_name
                    key = (src_name, actual_pred, tgt_name)
                    if key in graph_relation_dict:
                        continue
                    graph_relation_dict[key] = RelationBelief(
                        source=src_name,
                        predicate=actual_pred,
                        target=tgt_name,
                        confidence=edge.confidence,
                        source_modality=edge.modality,
                        evidence_ids=[oid for oid in (edge.observation_id,) if oid],
                        polarity=edge.polarity,
                    )

        # 3. Merge current relations (override graph for same triple)
        for r in current_relations:
            key = (r["source"], r["predicate"], r["target"])
            conf = r.get("confidence", 0.5)
            if key in graph_relation_dict:
                existing = graph_relation_dict[key]
                existing.confidence = max(existing.confidence, conf)
                oid = r.get("observation_id", "")
                if oid and oid not in existing.evidence_ids:
                    existing.evidence_ids.append(oid)
            else:
                graph_relation_dict[key] = RelationBelief(
                    source=r["source"],
                    predicate=r["predicate"],
                    target=r["target"],
                    confidence=conf,
                    source_modality=r.get("modality", "text"),
                    evidence_ids=[oid for oid in (r.get("observation_id", ""),) if oid],
                    polarity=r.get("polarity", True),
                )

        relations = list(graph_relation_dict.values())

        # 4. Extract attributes from property-style relations
        attributes: dict[str, dict[str, float]] = {}
        for r in relations:
            if r.predicate in self._PROPERTY_PREDICATES:
                attributes.setdefault(r.source, {})[r.target] = r.confidence

        # 5. Detect competing hypotheses (different predicate, same s/t pair)
        by_pair: dict[tuple[str, str], list[RelationBelief]] = {}
        for r in relations:
            by_pair.setdefault((r.source, r.target), []).append(r)

        alternatives: list[WorldState] = []
        for _pair, rels in by_pair.items():
            if len(rels) > 1:
                for alt_rel in rels:
                    alt = WorldState(
                        entities=dict(entities),
                        relations=[alt_rel],
                        attributes=dict(attributes),
                    )
                    alternatives.append(alt)
                    break

        state = WorldState(
            entities=entities,
            relations=relations,
            attributes=attributes,
            alternative_hypotheses=alternatives,
        )
        self._history.append(state)
        return state

    @property
    def last_state(self) -> WorldState | None:
        return self._history[-1] if self._history else None
