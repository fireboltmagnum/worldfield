"""Decoder abstraction — generates natural language from world state.

Supports multiple backends:
- ``template`` — rule-based fallback (always available)
- ``transformers`` — uses HuggingFace ``transformers`` library when installed
"""
from __future__ import annotations

from typing import Any

from .serializer import StateSerializer
from .prompts import PromptConstructor


class DecoderError(RuntimeError):
    """Raised when text generation fails."""


class Decoder:
    """Abstract language decoder.

    Parameters
    ----------
    backend:
        ``"template"`` — rule-based (always works)
        ``"transformers"`` — HuggingFace model (requires ``transformers``)
    model_name:
        HuggingFace model name (only used for ``"transformers"`` backend).
    device:
        Torch device string (``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        backend: str = "template",
        model_name: str = "google/flan-t5-small",
        device: str = "cpu",
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.serializer = StateSerializer()
        self.prompts = PromptConstructor()
        self._model = None
        self._tokenizer = None

        if backend == "transformers":
            self._lazy_load()

    def _lazy_load(self):
        """Load the HuggingFace model on first use (if backend is transformers)."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, device_map=self.device
            )
        except ImportError:
            raise DecoderError(
                "transformers backend requires the `transformers` package. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise DecoderError(f"Failed to load model {self.model_name}: {e}")

    def generate(
        self,
        world_state: dict[str, Any] | None = None,
        inference_result: dict[str, Any] | None = None,
        user_input: str = "",
    ) -> str:
        """Generate a natural-language response from the current cognitive state.

        Parameters
        ----------
        world_state:
            Dict from :meth:`WorldState.to_dict()`.
        inference_result:
            Dict from :meth:`InferenceResult.to_dict()`.
        user_input:
            The user's original text (optional).

        Returns
        -------
        str
            Generated response text.
        """
        serialized = self.serializer.serialize(world_state, inference_result)

        if self.backend == "transformers":
            return self._generate_with_model(serialized, user_input)
        else:
            return self._generate_with_template(serialized, user_input)

    def _generate_with_model(self, serialized: str, user_input: str) -> str:
        """Generate using HuggingFace transformers model."""
        if self._model is None:
            self._lazy_load()
        prompt = self.prompts.build(serialized, user_input)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generate_with_template(self, serialized: str, user_input: str) -> str:
        """Rule-based fallback that always works without ML dependencies."""
        lines: list[str] = []

        if user_input:
            lines.append(f"I understand you said: \"{user_input}\".")
            lines.append("")

        # Parse the serialized text to build a response
        has_entities = "Entities:" in serialized
        has_relations = "Relations:" in serialized
        has_inferences = "Inferences:" in serialized
        has_contradictions = "Contradictions:" in serialized

        if not has_entities and not has_relations:
            lines.append(
                "I don't have much to go on yet. "
                "Tell me something to help me build my understanding."
            )
        else:
            entities = self._extract_bullets(serialized, "Entities:")
            relations = self._extract_bullets(serialized, "Relations:")
            inferences = self._extract_bullets(serialized, "Inferences:")
            contradictions = self._extract_bullets(serialized, "Contradictions:")

            if entities:
                names = [e.split("(")[0].strip("- ").strip() for e in entities[:3]]
                lines.append(
                    f"I can see {', '.join(names)} "
                    f"in my current world model."
                )

            if relations:
                key_rel = relations[0]
                lines.append(f"I observe that {key_rel.strip('- ')}.")

            if inferences:
                for inv in inferences[:2]:
                    lines.append(f"I infer that {inv.strip('- ')}.")

            if contradictions:
                lines.append(
                    "I notice some competing interpretations: "
                    f"{contradictions[0].strip('- ')}."
                )

        return " ".join(lines)

    @staticmethod
    def _extract_bullets(text: str, section: str) -> list[str]:
        """Extract bullet-point lines under a section heading."""
        lines = text.split("\n")
        collecting = False
        bullets: list[str] = []
        for line in lines:
            if line.strip().startswith(section.rstrip(":")):
                collecting = True
                continue
            if collecting:
                if line.strip().startswith("-"):
                    bullets.append(line.strip())
                elif line.strip() == "":
                    continue
                elif not line.strip().startswith("-") and bullets:
                    break
        return bullets
