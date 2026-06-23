"""Small shared config objects for experiment runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Common knobs used by lightweight experiment wrappers."""

    seed: int = 0
    n_events: int = 4000
    report_subdir: str = "reports"
