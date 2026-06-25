"""Simulation — predicts possible futures from the current world state.

The simulator takes the current :class:`~worldfield.core.world_state.WorldState`
and predicts what *could* happen next based on the relations and entities
in the state. This is the beginning of :emphasis:`prediction`, one of the
core ingredients of intelligence.
"""
from __future__ import annotations

from typing import Any

from ..core.world_graph import WorldGraph


class Simulator:
    """Predict possible futures from the current world state.

    Parameters
    ----------
    graph:
        The knowledge graph used for lookups during simulation.
    """

    def __init__(self, graph: WorldGraph):
        self.graph = graph

    def simulate(
        self,
        world_state: dict[str, Any] | None = None,
        actions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Simulate possible outcomes given the current state.

        Parameters
        ----------
        world_state:
            Dict from :meth:`WorldState.to_dict()`.
        actions:
            Optional list of actions to simulate (e.g. ``"cat jumps"``).
            If empty, derives possible actions from the current state.

        Returns
        -------
        list[dict]
            Each dict represents one possible future outcome with
            ``"description"``, ``"probability"`` and ``"changes"`` keys.
        """
        if not world_state:
            return []

        entities = world_state.get("entities", {})
        relations = world_state.get("relations", [])
        outcomes: list[dict[str, Any]] = []

        actions_to_simulate = actions or self._derive_actions(
            entities, relations
        )

        for action in actions_to_simulate:
            outcome = self._simulate_one(action, entities, relations)
            if outcome:
                outcomes.append(outcome)

        outcomes.sort(key=lambda o: -o.get("probability", 0.0))
        return outcomes

    def predict(
        self,
        entity: str,
        predicate: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict likely relations for an entity based on graph knowledge.

        Parameters
        ----------
        entity:
            The entity name.
        predicate:
            Optional predicate to constrain predictions.

        Returns
        -------
        list[dict]
            Predicted relations with confidence.
        """
        node = self.graph.get_concept(entity)
        if node is None:
            return []

        predictions: list[dict[str, Any]] = []
        seen_targets: set[str] = set()

        for edge in self.graph.edges:
            if edge.source_id == node.id:
                tgt = self.graph.nodes.get(edge.target_id)
                if tgt is None or tgt.canonical_name in seen_targets:
                    continue
                if predicate and edge.predicate != predicate:
                    continue
                seen_targets.add(tgt.canonical_name)
                predictions.append({
                    "entity": entity,
                    "predicate": edge.predicate,
                    "target": tgt.canonical_name,
                    "confidence": edge.confidence,
                    "type": "existing",
                })

            if edge.target_id == node.id:
                src = self.graph.nodes.get(edge.source_id)
                if src is None or src.canonical_name in seen_targets:
                    continue
                if predicate and edge.predicate != predicate:
                    continue
                seen_targets.add(src.canonical_name)
                predictions.append({
                    "entity": entity,
                    "predicate": f"~{edge.predicate}",
                    "target": src.canonical_name,
                    "confidence": edge.confidence,
                    "type": "existing",
                })

        return predictions

    # ── Internal ───────────────────────────────────────────────────────

    def _derive_actions(
        self,
        entities: dict[str, float],
        relations: list[dict[str, Any]],
    ) -> list[str]:
        """Derive possible actions from the current world state."""
        actions: list[str] = []

        # For each entity with a spatial relation, suggest movement
        for r in relations:
            src = r.get("source", "")
            tgt = r.get("target", "")
            pred = r.get("predicate", "")

            # Spatial: entity is on/in something → suggests leaving
            if pred in ("sat_on", "sleeping_on", "located_on", "located_in"):
                actions.append(f"{src} leaves {tgt}")

            # Entity is chased → suggests fleeing
            if pred == "chased" and src in entities:
                actions.append(f"{src} flees")
                actions.append(f"{src} hides")

            # Agent relation → suggests acting on target
            if pred in ("plays_with", "has"):
                actions.append(f"{src} interacts with {tgt}")

        # Generic actions for entities
        for name in entities:
            actions.append(f"{name} stays")
            actions.append(f"{name} moves")

        return actions[:8]  # limit

    def _simulate_one(
        self,
        action: str,
        entities: dict[str, float],
        relations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Simulate a single action and return the predicted outcome."""
        action_lower = action.lower()

        # Movement: entity leaves something
        if "leaves" in action_lower or "moves" in action_lower:
            entity = action.split()[0]
            return {
                "description": f"{entity} changes location",
                "probability": 0.3,
                "changes": [
                    {"entity": entity, "property": "location",
                     "old": "current", "new": "unknown"},
                ],
            }

        # Interaction
        if "interacts" in action_lower:
            entity = action.split()[0]
            return {
                "description": f"{entity} interacts with nearby object",
                "probability": 0.5,
                "changes": [
                    {"entity": entity, "property": "state",
                     "old": "idle", "new": "active"},
                ],
            }

        # Staying
        if "stays" in action_lower:
            entity = action.split()[0]
            return {
                "description": f"{entity} remains in place",
                "probability": 0.7,
                "changes": [],
            }

        # Fleeing
        if "flees" in action_lower:
            entity = action.split()[0]
            return {
                "description": f"{entity} flees from danger",
                "probability": 0.6,
                "changes": [
                    {"entity": entity, "property": "location",
                     "old": "current", "new": "away"},
                    {"entity": entity, "property": "state",
                     "old": "calm", "new": "alert"},
                ],
            }

        if "hides" in action_lower:
            entity = action.split()[0]
            return {
                "description": f"{entity} hides",
                "probability": 0.4,
                "changes": [
                    {"entity": entity, "property": "visibility",
                     "old": "visible", "new": "hidden"},
                ],
            }

        return None
