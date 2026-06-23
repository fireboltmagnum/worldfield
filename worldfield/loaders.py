"""Shared path helpers for checkpoints and generated artifacts."""

from __future__ import annotations

from pathlib import Path

from .project import ROOT


ARTIFACT_ROOT = ROOT / "reports" / "artifacts"


def artifact_dir(day_name: str) -> Path:
    """Return the artifact directory for one experiment day."""
    return ARTIFACT_ROOT / day_name


def checkpoint_path(name: str, day_name: str = "day_one") -> Path:
    """Return a checkpoint path under ``reports/artifacts``."""
    return artifact_dir(day_name) / name
