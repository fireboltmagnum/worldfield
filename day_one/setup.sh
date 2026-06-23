#!/usr/bin/env bash
# One-time setup: create a Python 3.12 venv and install deps.
# torch has no stable wheels for Python 3.14 yet, so we pin to 3.12.
set -e
cd "$(dirname "$0")"

PY=$(command -v python3.12 || true)
if [ -z "$PY" ]; then
  echo "Python 3.12 not found. Install it (e.g. 'brew install python@3.12') and re-run."
  exit 1
fi

echo "Creating venv with $PY ..."
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Now run:"
echo "  source day_one/.venv/bin/activate"
echo "  python day_one/train.py"
