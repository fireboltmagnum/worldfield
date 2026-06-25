"""Prompt constructor — wraps serialised state into a prompt for a language decoder."""
from __future__ import annotations

from typing import Literal

ResponseStyle = Literal["concise", "detailed", "question"]


class PromptConstructor:
    """Build prompts from serialised world state for the decoder model.

    Parameters
    ----------
    system_prompt:
        Optional system-level instruction prepended to every prompt.
    """

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or self._default_system()

    @staticmethod
    def _default_system() -> str:
        return (
            "You are a cognitive system. You maintain a world model "
            "and reason about what you observe. "
            "Respond naturally based on your current understanding."
        )

    def build(
        self,
        serialized_state: str,
        user_input: str = "",
        style: ResponseStyle = "concise",
    ) -> str:
        """Wrap serialised state into a full prompt.

        Parameters
        ----------
        serialized_state:
            Output of :meth:`StateSerializer.serialize`.
        user_input:
            The user's original input (if any).
        style:
            ``"concise"`` — short response (default)
            ``"detailed"`` — thorough response
            ``"question"`` — user asked a specific question
        """
        style_instruction = self._style_instruction(style)
        parts = [self.system_prompt, "", style_instruction]

        if user_input:
            parts.append(f"User said: {user_input}")

        parts.append("Current world state:")
        parts.append(serialized_state)
        parts.append("Generate a response:")
        return "\n".join(parts)

    def build_simple(
        self,
        serialized_state: str,
        user_input: str = "",
    ) -> str:
        """Simpler prompt without style control (for template fallback)."""
        parts = []
        if user_input:
            parts.append(f"Input: {user_input}")
        parts.append("Current understanding:")
        parts.append(serialized_state)
        parts.append("Response:")
        return "\n".join(parts)

    @staticmethod
    def _style_instruction(style: ResponseStyle) -> str:
        instructions = {
            "concise": (
                "Be concise. One or two sentences describing what you "
                "understand and any new conclusions."
            ),
            "detailed": (
                "Be thorough. Describe what you observe, what you already "
                "knew, what you inferred, and any contradictions."
            ),
            "question": (
                "Answer the user's question based on your current "
                "understanding. Mention your confidence level."
            ),
        }
        return instructions.get(style, instructions["concise"])
