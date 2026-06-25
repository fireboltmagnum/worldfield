"""Planning — decomposes goals into executable steps.

The planner takes a :class:`~worldfield.core.goals.Goal` and the current
:class:`~worldfield.core.world_state.WorldState` and produces a sequence
of steps. If a step fails the planner can replan around it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """A single atomic step in a plan.

    Parameters
    ----------
    action:
        What to do (e.g. ``"build"``, ``"test"``, ``"query"``).
    target:
        What the action applies to (e.g. ``"serializer"``).
    status:
        ``"pending"``, ``"in_progress"``, ``"completed"``, ``"failed"``.
    dependencies:
        Other step descriptions this step depends on.
    description:
        Human-readable summary.
    """

    action: str
    target: str
    status: str = "pending"
    dependencies: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.action} {self.target}"


class Planner:
    """Decompose goals into executable steps.

    Uses a simple rule-based approach: for known action patterns it
    generates the appropriate sub-steps. Unknown goals produce a single
    generic step.
    """

    _KNOWN_PATTERNS: dict[str, list[str]] = {
        "build": ["design", "implement", "test", "benchmark"],
        "implement": ["create_stub", "add_logic", "add_tests", "verify"],
        "design": ["research", "sketch_architecture", "review"],
        "test": ["write_tests", "run_tests", "fix_failures"],
        "learn": ["observe", "extract_patterns", "update_model"],
        "analyze": ["gather_data", "process", "summarize"],
        "query": ["resolve_concept", "traverse_graph", "format_result"],
    }

    def __init__(self):
        self._plan_history: list[list[PlanStep]] = []

    def plan(
        self,
        goal_description: str,
        world_state_hint: dict[str, Any] | None = None,
    ) -> list[PlanStep]:
        """Decompose a goal into steps.

        Parameters
        ----------
        goal_description:
            The goal text, e.g. ``"Build decoder"`` or ``"test serializer"``.
        world_state_hint:
            Optional world state dict (reserved for future use).

        Returns
        -------
        list[PlanStep]
            Ordered steps to execute.
        """
        goal_lower = goal_description.lower()
        steps: list[PlanStep] = []

        # Match known action patterns
        for verb, substeps in self._KNOWN_PATTERNS.items():
            if verb in goal_lower:
                # Extract the target (everything after the verb)
                target = goal_lower.replace(verb, "").strip().lstrip()
                # Remove articles
                for art in ("the ", "a ", "an "):
                    if target.startswith(art):
                        target = target[len(art):]
                if not target:
                    target = verb

                prev = ""
                for i, substep in enumerate(substeps):
                    deps = [prev] if prev else []
                    step = PlanStep(
                        action=substep,
                        target=target,
                        dependencies=deps,
                        description=f"{substep} {target}",
                    )
                    steps.append(step)
                    prev = step.description
                break

        # Fallback: single generic step
        if not steps:
            steps.append(PlanStep(
                action="process",
                target=goal_description,
                description=f"process {goal_description}",
            ))

        self._plan_history.append(steps)
        return steps

    def replan(
        self,
        failed_step: PlanStep,
        remaining: list[PlanStep],
    ) -> list[PlanStep]:
        """Replan after a step fails.

        Marks the failed step and regenerates successor steps as new
        plan attempts.

        Parameters
        ----------
        failed_step:
            The step that failed.
        remaining:
            Steps that were scheduled after the failure.

        Returns
        -------
        list[PlanStep]
            Replacement steps.
        """
        failed_step.status = "failed"

        # Generate a fix step and retry
        fix = PlanStep(
            action="fix",
            target=failed_step.description,
            description=f"fix {failed_step.description}",
            dependencies=[failed_step.description],
        )
        retry = PlanStep(
            action="retry",
            target=failed_step.description,
            description=f"retry {failed_step.description}",
            dependencies=[fix.description],
        )
        return [fix, retry] + remaining

    def get_summary(self) -> dict[str, Any]:
        return {
            "n_plans": len(self._plan_history),
            "latest_plan":
                [
                    {"action": s.action, "target": s.target, "status": s.status}
                    for s in (self._plan_history[-1] if self._plan_history else [])
                ]
            ,
        }
