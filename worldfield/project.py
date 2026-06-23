"""Project-level metadata and lightweight setup helpers.

The experiment scripts intentionally remain in their day-by-day folders. This
module gives tests, docs, and future scripts one stable place to ask basic
questions about the repository without importing heavy ML dependencies.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DAY_DIRECTORIES = (
    "day_one",
    "day_two",
    "day_three",
    "day_four",
    "day_five",
    "day_six",
    "day_seven",
    "day_eight",
    "day_nine",
)

_PACKAGE_IMPORTS = {
    "torch": "torch",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "faiss-cpu": "faiss",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
}


def required_packages() -> dict[str, bool]:
    """Return whether each repo-level dependency is importable."""
    return {name: find_spec(module) is not None for name, module in _PACKAGE_IMPORTS.items()}
