"""Context Layer — maintains conversational context across turns.

Without context every input is processed independently. The context layer
tracks the current topic, recent concepts, and interaction history so that
:emphasis:`the system knows what we are talking about`.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any


class ContextManager:
    """Tracks the current conversational context across turns.

    Parameters
    ----------
    max_history:
        Number of recent turns kept in history.
    max_recent_concepts:
        Sliding-window size for recently mentioned concepts.
    """

    def __init__(
        self,
        max_history: int = 10,
        max_recent_concepts: int = 20,
    ):
        self.topic: str = ""
        self.recent_concepts: deque[str] = deque(maxlen=max_recent_concepts)
        self.history: list[dict[str, Any]] = []
        self._max_history = max_history
        self._turn_count = 0

    # ── Public API ─────────────────────────────────────────────────────

    def update(
        self,
        user_input: str,
        entities: list[str],
        activated: list[tuple[str, float]],
        generated_text: str = "",
    ) -> None:
        """Update context after processing one input turn.

        Parameters
        ----------
        user_input:
            Raw user text.
        entities:
            Extracted concept names from this turn.
        activated:
            ``[(name, activation), ...]`` from the activation engine.
        generated_text:
            The system's generated response (if any).
        """
        self._turn_count += 1

        # Update recent concepts (sliding window)
        for name, _ in activated:
            if name not in self.recent_concepts:
                self.recent_concepts.append(name)

        # Add input entities that aren't tracked
        for name in entities:
            if name not in self.recent_concepts:
                self.recent_concepts.append(name)

        # Infer topic: first extracted entity, else keep current topic
        if entities:
            self.topic = entities[0]

        # Store interaction in history
        entry: dict[str, Any] = {
            "turn": self._turn_count,
            "input": user_input,
            "entities": list(entities),
            "response": generated_text,
            "timestamp": time.time(),
        }
        self.history.append(entry)
        if len(self.history) > self._max_history:
            self.history.pop(0)

    def get_context_summary(self) -> dict[str, Any]:
        """Return a structured summary of the current context."""
        return {
            "topic": self.topic,
            "recent_concepts": list(self.recent_concepts),
            "history_length": len(self.history),
            "turn": self._turn_count,
        }

    def format_context_block(self) -> str:
        """Render context as text for the NLG decoder."""
        parts = [f"Topic: {self.topic}"]
        if self.recent_concepts:
            parts.append(
                "Recent concepts: " + ", ".join(list(self.recent_concepts)[-8:])
            )
        return "\n".join(parts)

    def state_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "recent_concepts": list(self.recent_concepts),
            "history": self.history,
            "turn": self._turn_count,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.topic = sd.get("topic", "")
        self.recent_concepts = deque(
            sd.get("recent_concepts", []),
            maxlen=self.recent_concepts.maxlen,
        )
        self.history = sd.get("history", [])
        self._turn_count = sd.get("turn", 0)

    def reset(self) -> None:
        self.topic = ""
        self.recent_concepts.clear()
        self.history.clear()
        self._turn_count = 0
