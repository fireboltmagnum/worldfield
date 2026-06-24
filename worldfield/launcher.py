"""Bootstrap launcher — checks deps before running worldfield."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import warnings


_REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "sentence_transformers",
    "spacy",
    "chromadb",
    "rich",
    "prompt_toolkit",
    "PIL",
]

_SPACY_MODEL = "en_core_web_sm"


def _check_python() -> None:
    if sys.version_info < (3, 10):
        print("worldfield requires Python >= 3.10")
        sys.exit(1)


def _check_packages() -> None:
    missing = []
    for pkg in _REQUIRED_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if not missing:
        return
    print(f"Installing {len(missing)} missing package(s): {', '.join(missing)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("Done.")


def _check_spacy_model() -> None:
    try:
        import spacy
        spacy.load(_SPACY_MODEL)
    except OSError:
        print("Downloading spaCy language model (first run)...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", _SPACY_MODEL],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("Done.")
    except ImportError:
        pass


def check_deps():
    """Public entry point: run all checks."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    _check_python()
    _check_packages()
    _check_spacy_model()


def main():
    check_deps()
    from .__main__ import main as cli_main
    cli_main()
