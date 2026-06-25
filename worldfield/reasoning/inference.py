"""Inference Engine — reasoning over the current world state.

The inference engine takes the current :class:`~worldfield.core.world_state.WorldState`
and produces new conclusions through property inheritance, relation composition,
and contradiction detection. Every inference maintains a chain of explanation steps
so the system can explain *why* it believes something.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.world_graph import WorldGraph
from ..core.world_state import WorldState, RelationBelief
from .graph_ops import GraphOps


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class InferenceStep:
    """A single atomic step in an inference chain."""
    rule: str
    premise: str
    confidence: float


@dataclass
class Inference:
    """A conclusion derived through inference.

    Parameters
    ----------
    conclusion:
        Human-readable description, e.g. ``"cat has fur"``.
    rule:
        The rule that produced this inference
        (``"property_inheritance"``, ``"relation_composition"``).
    confidence:
        Combined confidence along the inference chain.
    source:
        Source entity.
    predicate:
        Predicate of the inferred relation.
    target:
        Target entity.
    steps:
        The chain of reasoning steps.
    novel:
        *True* if this is a genuinely new inference (not a known fact).
    """

    conclusion: str
    rule: str
    confidence: float
    source: str
    predicate: str
    target: str
    steps: list[InferenceStep] = field(default_factory=list)
    novel: bool = True


@dataclass
class Contradiction:
    """A detected contradiction in the current world state.

    Parameters
    ----------
    entity:
        The entity involved.
    predicate_a:
        First conflicting predicate.
    predicate_b:
        Second conflicting predicate.
    target:
        The common target or concept involved.
    confidence_a:
        Confidence of the first belief.
    confidence_b:
        Confidence of the second belief.
    description:
        Human-readable description.
    """

    entity: str
    predicate_a: str
    predicate_b: str
    target: str
    confidence_a: float
    confidence_b: float
    description: str = ""


@dataclass
class InferenceResult:
    """The full output of a reasoning pass over a world state."""

    inferences: list[Inference] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def n_inferences(self) -> int:
        return len(self.inferences)

    @property
    def n_contradictions(self) -> int:
        return len(self.contradictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inferences": [
                {
                    "conclusion": i.conclusion,
                    "rule": i.rule,
                    "confidence": i.confidence,
                    "source": i.source,
                    "predicate": i.predicate,
                    "target": i.target,
                    "steps": [
                        {"rule": s.rule, "premise": s.premise, "confidence": s.confidence}
                        for s in i.steps
                    ],
                    "novel": i.novel,
                }
                for i in self.inferences
            ],
            "contradictions": [
                {
                    "entity": c.entity,
                    "predicate_a": c.predicate_a,
                    "predicate_b": c.predicate_b,
                    "target": c.target,
                    "confidence_a": c.confidence_a,
                    "confidence_b": c.confidence_b,
                    "description": c.description,
                }
                for c in self.contradictions
            ],
            "processing_time_ms": self.processing_time_ms,
        }


# ── Inference Engine ──────────────────────────────────────────────────

class InferenceEngine:
    """Reason over a :class:`WorldState` to produce new conclusions.

    Parameters
    ----------
    graph:
        The long-term knowledge graph used for property lookups
        (e.g. ``is_a`` hierarchies).
    max_inheritance_depth:
        How many ``is_a`` hops to follow during property inheritance.
    composition_predicates:
        Which predicate pairs are eligible for composition.
    """

    _INHERITABLE_PREDICATES = frozenset({
        "has", "has_property", "has_attribute", "can", "is",
    })

    def __init__(
        self,
        graph: WorldGraph,
        max_inheritance_depth: int = 3,
    ):
        self.graph = graph
        self.ops = GraphOps(graph)
        self.max_inheritance_depth = max_inheritance_depth

    # ── Public API ─────────────────────────────────────────────────────

    def reason(self, world_state: WorldState) -> InferenceResult:
        """Run all inference strategies on the current world state.

        Returns
        -------
        InferenceResult
            Inferences, contradictions, and timing.
        """
        t0 = time.perf_counter()
        inferences: list[Inference] = []
        contradictions: list[Contradiction] = []

        inferences.extend(self._inherit_properties(world_state))
        inferences.extend(self._compose_relations(world_state))
        contradictions.extend(self._detect_contradictions(world_state))

        elapsed = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            inferences=inferences,
            contradictions=contradictions,
            processing_time_ms=elapsed,
        )

    def explain(self, inference: Inference) -> str:
        """Render an inference as a human-readable explanation string."""
        parts = [f"∵ {inference.conclusion}"]
        for s in inference.steps:
            parts.append(f"  {s.rule}: {s.premise}  (conf={s.confidence:.2f})")
        return "\n".join(parts)

    # ── Property Inheritance ───────────────────────────────────────────

    def _inherit_properties(self, ws: WorldState) -> list[Inference]:
        """If ``X is_a Y`` and ``Y has P`` then ``X has P``."""
        inferences: list[Inference] = []

        # Collect (entity, is_a_ancestor) pairs from the world state
        is_a_chain: dict[str, set[str]] = {}
        for r in ws.relations:
            if r.predicate == "is_a" and r.polarity:
                is_a_chain.setdefault(r.source, set()).add(r.target)

        for entity in ws.entities:
            ancestors = list(is_a_chain.get(entity, []))
            # Also query graph for transitive is_a ancestors
            graph_ancestors = self.ops.transitive_closure(
                entity, "is_a", max_hops=self.max_inheritance_depth
            )
            ancestors.extend(a for a in graph_ancestors if a not in ancestors)
            ancestors_set = set(ancestors)

            for ancestor in ancestors_set:
                # Find properties of this ancestor in the graph
                for edge in self.graph.get_relations(ancestor):
                    if edge.predicate not in self._INHERITABLE_PREDICATES:
                        continue
                    tgt = self.graph.nodes.get(edge.target_id)
                    if tgt is None:
                        continue

                    # Check if entity already has this property in WS
                    existing = ws.get_belief(entity, edge.predicate, tgt.canonical_name)
                    if existing is not None:
                        continue

                    combined_conf = edge.confidence * 0.9  # discount for inheritance
                    conclusion = f"{entity} {edge.predicate} {tgt.canonical_name}"
                    steps = [
                        InferenceStep(
                            rule="is_a_lookup",
                            premise=f"{entity} is_a {ancestor}",
                            confidence=0.9,
                        ),
                        InferenceStep(
                            rule="property_lookup",
                            premise=f"{ancestor} {edge.predicate} {tgt.canonical_name}",
                            confidence=edge.confidence,
                        ),
                    ]
                    inferences.append(Inference(
                        conclusion=conclusion,
                        rule="property_inheritance",
                        confidence=combined_conf,
                        source=entity,
                        predicate=edge.predicate,
                        target=tgt.canonical_name,
                        steps=steps,
                        novel=True,
                    ))

        return inferences

    # ── Relation Composition ───────────────────────────────────────────

    def _compose_relations(self, ws: WorldState) -> list[Inference]:
        """If ``X -[R1]→ Y`` and ``Y -[R2]→ Z`` then ``X -[R1∘R2]→ Z``.

        Operates over the combined set of world-state relations + graph
        neighbors of entities in the world state.
        """
        inferences: list[Inference] = []

        # Build adjacency from world state relations
        adj: dict[str, list[tuple[str, str, float]]] = {}  # source -> [(pred, target, conf)]
        for r in ws.relations:
            adj.setdefault(r.source, []).append((r.predicate, r.target, r.confidence))
            adj.setdefault(r.target, []).append((r.predicate, r.source, r.confidence))

        # Extend with graph neighbours for entities in the WS
        for entity in ws.entities:
            node = self.graph.get_concept(entity)
            if node is None:
                continue
            for edge in self.graph.get_relations(entity):
                tgt = self.graph.nodes.get(edge.target_id)
                if tgt is None:
                    continue
                adj.setdefault(entity, []).append(
                    (edge.predicate, tgt.canonical_name, edge.confidence)
                )

        composed_rules: dict[str, str] = {
            "located_on": "spatial_containment",
            "located_in": "spatial_containment",
            "part_of": "meronomy",
            "is_a": "taxonomy",
        }

        seen_pairs: set[tuple[str, str, str]] = set()
        for entity, rels in adj.items():
            for pred_a, mid, conf_a in rels:
                if mid not in adj:
                    continue
                for pred_b, z, conf_b in adj[mid]:
                    key = (entity, f"{pred_a}/{pred_b}", z)
                    if key in seen_pairs:
                        continue
                    if z == entity:
                        continue  # skip reflexive cycles
                    seen_pairs.add(key)
                    combined_conf = conf_a * conf_b
                    rule_name = composed_rules.get(pred_a, "relay")
                    conclusion = f"{entity} {pred_a}/{pred_b} {z}"
                    steps = [
                        InferenceStep(
                            rule=rule_name,
                            premise=f"{entity} -[{pred_a}]→ {mid}",
                            confidence=conf_a,
                        ),
                        InferenceStep(
                            rule=rule_name,
                            premise=f"{mid} -[{pred_b}]→ {z}",
                            confidence=conf_b,
                        ),
                    ]
                    inferences.append(Inference(
                        conclusion=conclusion,
                        rule="relation_composition",
                        confidence=combined_conf,
                        source=entity,
                        predicate=f"{pred_a}/{pred_b}",
                        target=z,
                        steps=steps,
                        novel=True,
                    ))

        return inferences

    # ── Contradiction Detection ────────────────────────────────────────

    def _detect_contradictions(self, ws: WorldState) -> list[Contradiction]:
        """Detect conflicting beliefs in the world state.

        A contradiction occurs when two relations share the same source-target
        pair but have different predicates that are semantically incompatible
        (e.g. ``sleeping_on`` vs ``sitting_on`` for the same entity-pair).
        """
        contradictions: list[Contradiction] = []

        by_pair: dict[tuple[str, str], list[RelationBelief]] = {}
        for r in ws.relations:
            by_pair.setdefault((r.source, r.target), []).append(r)

        for (src, tgt), rels in by_pair.items():
            if len(rels) < 2:
                continue
            # Multiple predicates for the same source-target pair
            preds = [r.predicate for r in rels]
            if len(set(preds)) < 2:
                continue  # same predicate, different evidence — not a contradiction

            # Sort by confidence descending; the top is the "primary" belief
            sorted_rels = sorted(rels, key=lambda r: -r.confidence)
            primary = sorted_rels[0]
            for alt in sorted_rels[1:]:
                contradictions.append(Contradiction(
                    entity=src,
                    predicate_a=primary.predicate,
                    predicate_b=alt.predicate,
                    target=tgt,
                    confidence_a=primary.confidence,
                    confidence_b=alt.confidence,
                    description=(
                        f"{src} -[{primary.predicate}]→ {tgt} "
                        f"vs {src} -[{alt.predicate}]→ {tgt}"
                    ),
                ))

        return contradictions
