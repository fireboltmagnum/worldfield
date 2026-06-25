"""Goal Layer — maintains the system's objectives across turns.

Without goals the system reacts but does not act — it thinks, forgets,
thinks, forgets. The goal layer gives the system direction by tracking
primary goals, subgoals, current tasks, blockers, and progress.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Goal:
    """A single objective the system is working toward.

    Parameters
    ----------
    description:
        Human-readable description, e.g. ``"Build reasoning engine"``.
    priority:
        Higher values = higher priority.
    status:
        ``"active"``, ``"completed"``, ``"blocked"``, or ``"cancelled"``.
    subgoals:
        Child goals that decompose this goal.
    blockers:
        Descriptions of what is blocking progress.
    progress:
        0.0–1.0 fraction complete.
    created_at:
        Seconds since epoch.
    completed_at:
        Seconds since epoch when completed.
    """

    description: str
    priority: int = 0
    status: str = "active"
    subgoals: list[Goal] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    progress: float = 0.0
    created_at: float = 0.0
    completed_at: float | None = None

    def __hash__(self) -> int:
        return id(self)


class GoalManager:
    """Manages the system's goal stack and current task tracking.

    Usage::

        gm = GoalManager()
        g = gm.add_goal("Build reasoning engine", priority=10)
        gm.add_subgoal(g, "Implement inheritance")
        gm.block(g, "Missing GraphOps")
        gm.verify()  # check if current task is complete
    """

    def __init__(self):
        self.goals: list[Goal] = []
        self._current_task: str = ""

    # ── Public API ─────────────────────────────────────────────────────

    def add_goal(
        self, description: str, priority: int = 0
    ) -> Goal:
        """Add a new top-level goal."""
        goal = Goal(
            description=description,
            priority=priority,
            created_at=time.time(),
        )
        self.goals.append(goal)
        self.goals.sort(key=lambda g: -g.priority)
        self._update_current_task()
        return goal

    def add_subgoal(self, parent: Goal, description: str) -> Goal:
        """Add a subgoal to an existing goal."""
        sub = Goal(
            description=description,
            priority=parent.priority - 1,
            created_at=time.time(),
        )
        parent.subgoals.append(sub)
        self._update_current_task()
        return sub

    def complete(self, goal: Goal) -> None:
        """Mark a goal as completed."""
        goal.status = "completed"
        goal.progress = 1.0
        goal.completed_at = time.time()
        self._update_current_task()

    def block(self, goal: Goal, reason: str) -> None:
        """Mark a goal as blocked by an external dependency."""
        goal.status = "blocked"
        if reason not in goal.blockers:
            goal.blockers.append(reason)
        self._update_current_task()

    def unblock(self, goal: Goal, reason: str) -> None:
        """Remove a blocker from a goal."""
        if reason in goal.blockers:
            goal.blockers.remove(reason)
        if not goal.blockers:
            goal.status = "active"
        self._update_current_task()

    def set_progress(self, goal: Goal, progress: float) -> None:
        """Update progress on a goal (0.0–1.0)."""
        goal.progress = max(0.0, min(1.0, progress))
        if goal.progress >= 1.0:
            self.complete(goal)

    def get_current_task(self) -> str:
        """Return the description of the most urgent active task."""
        return self._current_task

    def get_active_goals(self) -> list[Goal]:
        """Return all goals with status ``"active"``, sorted by priority."""
        return sorted(
            [g for g in self.goals if g.status == "active"],
            key=lambda g: -g.priority,
        )

    def verify(self) -> bool:
        """Check whether the current task/goal is verifiably complete.

        Returns *True* if all active goals have reached 100% progress.
        """
        active = self.get_active_goals()
        if not active:
            return True
        return all(g.progress >= 1.0 for g in active)

    def get_summary(self) -> dict[str, Any]:
        """Return a structured summary for display / NLG."""
        return {
            "n_goals": len(self.goals),
            "active": [g.description for g in self.get_active_goals()],
            "completed": [
                g.description for g in self.goals if g.status == "completed"
            ],
            "blocked": [
                {"goal": g.description, "blockers": g.blockers}
                for g in self.goals if g.status == "blocked"
            ],
            "current_task": self._current_task,
        }

    # ── Internal ───────────────────────────────────────────────────────

    def _update_current_task(self) -> None:
        """Set current task to the highest-priority active goal or subgoal."""
        for g in self.goals:
            if g.status == "active":
                # Check for active subgoals
                for sub in g.subgoals:
                    if sub.status == "active":
                        self._current_task = sub.description
                        return
                self._current_task = g.description
                return
        self._current_task = ""

    def state_dict(self) -> dict[str, Any]:
        def _goal_dict(g: Goal) -> dict:
            return {
                "description": g.description,
                "priority": g.priority,
                "status": g.status,
                "blockers": g.blockers,
                "progress": g.progress,
                "created_at": g.created_at,
                "completed_at": g.completed_at,
                "subgoals": [_goal_dict(s) for s in g.subgoals],
            }

        return {
            "goals": [_goal_dict(g) for g in self.goals],
            "current_task": self._current_task,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        def _from_dict(d: dict) -> Goal:
            return Goal(
                description=d["description"],
                priority=d.get("priority", 0),
                status=d.get("status", "active"),
                blockers=d.get("blockers", []),
                progress=d.get("progress", 0.0),
                created_at=d.get("created_at", 0.0),
                completed_at=d.get("completed_at"),
                subgoals=[_from_dict(s) for s in d.get("subgoals", [])],
            )

        self.goals = [_from_dict(g) for g in sd.get("goals", [])]
        self._current_task = sd.get("current_task", "")
        self._update_current_task()
