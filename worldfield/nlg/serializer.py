"""State serializer — converts WorldState + InferenceResult to structured text."""
from __future__ import annotations

from typing import Any


class StateSerializer:
    """Serialise a :class:`~worldfield.core.world_state.WorldState` and optional
    :class:`~worldfield.reasoning.inference.InferenceResult` into structured text
    that a language decoder can consume.

    Usage::

        serializer = StateSerializer()
        text = serializer.serialize(world_state_dict, inference_dict)
    """

    def serialize(
        self,
        world_state: dict[str, Any] | None = None,
        inference_result: dict[str, Any] | None = None,
    ) -> str:
        """Produce structured text from the current cognitive state."""
        lines: list[str] = []

        if world_state:
            entities = world_state.get("entities", {})
            relations = world_state.get("relations", [])
            attributes = world_state.get("attributes", {})

            if entities:
                lines.append("Entities:")
                for name, conf in sorted(entities.items(), key=lambda x: -x[1])[:10]:
                    attr = ""
                    if name in attributes:
                        attr_str = ", ".join(
                            f"{a} ({c:.2f})" for a, c in attributes[name].items()
                        )
                        attr = f"  [{attr_str}]"
                    lines.append(f"  - {name}  (conf={conf:.2f}){attr}")

            if relations:
                lines.append("")
                lines.append("Relations:")
                for r in relations[:10]:
                    src = r.get("source", "?")
                    pred = r.get("predicate", "?")
                    tgt = r.get("target", "?")
                    conf = r.get("confidence", 0.0)
                    pol = "" if r.get("polarity", True) else " NOT"
                    lines.append(
                        f"  - {src} -{pol}[{pred}]-> {tgt}  (conf={conf:.2f})"
                    )

        if inference_result:
            inferences = inference_result.get("inferences", [])
            contradictions = inference_result.get("contradictions", [])

            if inferences:
                lines.append("")
                lines.append("Inferences:")
                for inv in inferences[:5]:
                    conf = inv.get("confidence", 0.0)
                    src = inv.get("source", "?")
                    pred = inv.get("predicate", "?")
                    tgt = inv.get("target", "?")
                    lines.append(
                        f"  - {src} -[{pred}]-> {tgt}  (conf={conf:.2f})"
                    )

            if contradictions:
                lines.append("")
                lines.append("Contradictions:")
                for c in contradictions[:3]:
                    desc = c.get("description", "")
                    lines.append(f"  - {desc}")

        if not lines:
            lines.append("(empty state)")

        lines.append("")
        return "\n".join(lines)

    def serialize_graph_query(
        self, query_result: dict[str, Any]
    ) -> str:
        """Serialise a raw graph query result (used by the old query command)."""
        parts = query_result.get("related_concepts", [])
        if not parts:
            return "(no related concepts found)"
        lines = ["Related concepts:"]
        for p in parts[:10]:
            lines.append(
                f"  - {p.get('concept', '?')} "
                f"via {p.get('relation', '?')} "
                f"(conf={p.get('confidence', 0.0):.2f})"
            )
        return "\n".join(lines)
