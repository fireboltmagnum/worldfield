from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..core.world_graph import WorldGraph


_SPATIAL_PREDICATES = {
    "on", "in", "under", "near", "behind", "in front of",
    "above", "below", "beside", "next to", "beneath",
    "to the right of", "to the left of",
    "in front", "on top of", "inside", "outside",
}

_ATTRIBUTE_PREDICATES = {
    "has_attribute", "has_property", "has_color",
    "has_size", "has_shape", "has_texture",
    "is", "looks", "feels",
}

_TAXONOMY_PREDICATES = {"is_a", "kind_of", "type_of", "subclass_of"}

_COMPOSITION_PREDICATES = {"has", "has_part", "part_of", "contains", "consists_of"}

_BEHAVIOR_PREFIXES = ("ing", "ed")


def _is_behavior(pred: str) -> bool:
    return (
        pred.endswith("ing") or pred.endswith("ed")
    ) and pred not in _SPATIAL_PREDICATES and pred not in _ATTRIBUTE_PREDICATES


def _category(pred: str) -> str:
    pred_lower = pred.lower().strip()
    if pred_lower in _TAXONOMY_PREDICATES:
        return "taxonomy"
    if pred_lower in _ATTRIBUTE_PREDICATES:
        return "attribute"
    if pred_lower in _COMPOSITION_PREDICATES:
        return "composition"
    if pred_lower in _SPATIAL_PREDICATES:
        return "spatial"
    if _is_behavior(pred_lower):
        return "behavior"
    return "other"


@dataclass
class SummaryItem:
    value: str
    score: float
    count: int


@dataclass
class GroupSummary:
    label: str
    items: list[SummaryItem] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.items) == 0


@dataclass
class ConceptSummary:
    name: str
    total_observations: int
    avg_confidence: float
    groups: list[GroupSummary] = field(default_factory=list)


class ConceptSummarizer:
    def __init__(self, graph: WorldGraph):
        self.graph = graph

    def summarize(self, concept_name: str,
                  max_per_group: int = 8) -> ConceptSummary | None:
        node = self.graph.get_concept(concept_name)
        if node is None:
            return None

        # Collect all outgoing and incoming edges
        outgoing: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
        incoming: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)

        for edge in self.graph.edges:
            if edge.source_id == node.id and edge.target_id != node.id:
                best = outgoing[edge.predicate]
                key = self.graph.nodes.get(edge.target_id)
                if key and (key.canonical_name not in best
                            or edge.confidence > best[key.canonical_name][0]):
                    best[key.canonical_name] = (edge.confidence, edge.support_count)
            elif edge.target_id == node.id and edge.source_id != node.id:
                best = incoming[edge.predicate]
                src = self.graph.nodes.get(edge.source_id)
                if src and (src.canonical_name not in best
                            or edge.confidence > best[src.canonical_name][0]):
                    best[src.canonical_name] = (edge.confidence, edge.support_count)

        groups: list[GroupSummary] = []

        # 1. Taxonomy parents (outgoing is_a)
        tax_parents = [
            SummaryItem(value=v, score=s[0], count=s[1])
            for v, s in sorted(
                outgoing.get("is_a", {}).items(),
                key=lambda x: x[1][0], reverse=True,
            )[:max_per_group]
        ]
        if tax_parents:
            groups.append(GroupSummary(label="Core type", items=tax_parents))

        # 2. Attributes (outgoing has_attribute, is, etc.)
        attr_items: list[SummaryItem] = []
        for pred in _ATTRIBUTE_PREDICATES:
            for val, (conf, cnt) in outgoing.get(pred, {}).items():
                attr_items.append(SummaryItem(value=val, score=conf, count=cnt))
        attr_items.sort(key=lambda x: x.score, reverse=True)
        if attr_items:
            groups.append(GroupSummary(label="Attributes", items=attr_items[:max_per_group]))

        # 3. Behaviors (outgoing action verbs)
        behav_items: list[SummaryItem] = []
        for pred, targets in outgoing.items():
            if _is_behavior(pred):
                # For behaviors, include the predicate + target
                for obj, (conf, cnt) in targets.items():
                    behav_items.append(
                        SummaryItem(value=f"{pred} {obj}", score=conf, count=cnt)
                    )
        behav_items.sort(key=lambda x: x.score, reverse=True)
        if behav_items:
            groups.append(GroupSummary(label="Behaviors", items=behav_items[:max_per_group]))

        # 4. Spatial / common locations
        loc_items: list[SummaryItem] = []
        for pred in _SPATIAL_PREDICATES:
            for obj, (conf, cnt) in outgoing.get(pred, {}).items():
                loc_items.append(SummaryItem(value=f"{pred} {obj}", score=conf, count=cnt))
        loc_items.sort(key=lambda x: x.score, reverse=True)
        if loc_items:
            groups.append(GroupSummary(label="Common locations", items=loc_items[:max_per_group]))

        # 5. Things located near this concept (incoming spatial)
        in_loc_items: list[SummaryItem] = []
        for pred in _SPATIAL_PREDICATES:
            for subj, (conf, cnt) in incoming.get(pred, {}).items():
                in_loc_items.append(SummaryItem(value=f"{subj} {pred}", score=conf, count=cnt))
        in_loc_items.sort(key=lambda x: x.score, reverse=True)
        if in_loc_items:
            groups.append(GroupSummary(label="Objects nearby", items=in_loc_items[:max_per_group]))

        # 6. Taxonomy children (incoming is_a)
        children = [
            SummaryItem(value=v, score=s[0], count=s[1])
            for v, s in sorted(
                incoming.get("is_a", {}).items(),
                key=lambda x: x[1][0], reverse=True,
            )[:max_per_group]
        ]
        if children:
            groups.append(GroupSummary(label="Subtypes", items=children))

        # 7. Composition (outgoing has)
        has_items: list[SummaryItem] = []
        for pred in ("has", "has_part", "contains"):
            for obj, (conf, cnt) in outgoing.get(pred, {}).items():
                has_items.append(SummaryItem(value=obj, score=conf, count=cnt))
        has_items.sort(key=lambda x: x.score, reverse=True)
        if has_items:
            groups.append(GroupSummary(label="Parts", items=has_items[:max_per_group]))

        # 8. Other outgoing (uncategorized)
        other_out: list[SummaryItem] = []
        categorized_preds = (
            _TAXONOMY_PREDICATES | _ATTRIBUTE_PREDICATES
            | _COMPOSITION_PREDICATES | _SPATIAL_PREDICATES
            | {"has_attribute", "is_a", "has", "has_part", "part_of"}
        )
        for pred, targets in outgoing.items():
            if pred in categorized_preds or _is_behavior(pred):
                continue
            for obj, (conf, cnt) in targets.items():
                other_out.append(
                    SummaryItem(value=f"{pred} {obj}", score=conf, count=cnt)
                )
        other_out.sort(key=lambda x: x.score, reverse=True)
        if other_out:
            groups.append(GroupSummary(label="Other relations", items=other_out[:max_per_group]))

        total_obs = sum(
            s.count
            for g in groups for s in g.items
        )
        avg_conf = float(
            sum(s.score for g in groups for s in g.items)
            / max(sum(1 for g in groups for _ in g.items), 1)
        )

        return ConceptSummary(
            name=node.canonical_name,
            total_observations=total_obs,
            avg_confidence=round(avg_conf, 3),
            groups=groups,
        )
