"""ContextHorizon — consolidated view of what matters for the current cycle.

Created fresh each turn after Attention + Retrieval.
Activation only sees the horizon, not the full graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextHorizon:
    """Fixed-slot consolidation of everything relevant this cycle."""

    current_input: list[str] = field(default_factory=list)
    attended_concepts: list[Any] = field(default_factory=list)
    retrieved_memory: Any = None
    world_state: dict[str, float] = field(default_factory=dict)
    active_goals: list[Any] = field(default_factory=list)
    recent_reasoning: list[Any] = field(default_factory=list)

    def get_all_concepts(self) -> set[str]:
        results: set[str] = set()
        results.update(self.current_input)
        for sc in self.attended_concepts:
            if hasattr(sc, "name"):
                results.add(sc.name)
        if self.retrieved_memory is not None:
            results.update(self.retrieved_memory.nodes.keys())
        results.update(self.world_state.keys())
        for g in self.active_goals:
            if hasattr(g, "description"):
                # extract key nouns from description
                for word in g.description.split():
                    clean = word.strip(".,!?;:'\"()[]").lower()
                    if clean and len(clean) > 2:
                        results.add(clean)
        for r in self.recent_reasoning:
            if hasattr(r, "concepts"):
                results.update(r.concepts)
        return results

    def format_horizon_block(self) -> str:
        lines = ["=== Context Horizon ==="]
        if self.current_input:
            lines.append(f"Input: {' '.join(self.current_input)}")
        if self.attended_concepts:
            top = self.attended_concepts[:5]
            for sc in top:
                name = sc.name if hasattr(sc, "name") else str(sc)
                score = sc.score if hasattr(sc, "score") else 0
                lines.append(f"  {name} ({score:.2f})")
        if self.world_state:
            top_ws = sorted(self.world_state.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("World: " + ", ".join(f"{k}={v:.2f}" for k, v in top_ws))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_input": self.current_input,
            "attended_concepts": [
                {"name": s.name, "score": s.score, "signals": s.signals}
                if hasattr(s, "name") else str(s)
                for s in self.attended_concepts
            ],
            "retrieved_nodes": list(self.retrieved_memory.nodes.keys()) if self.retrieved_memory else [],
            "world_state": self.world_state,
            "n_goals": len(self.active_goals),
            "n_reasoning": len(self.recent_reasoning),
        }
