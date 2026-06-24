"""Persistence layer — saves/loads WorldField state to disk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Persistence:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

    # ── Slots ─────────────────────────────────────────────────────────

    def save_slots(self, state_dict: dict):
        self.db_path.mkdir(parents=True, exist_ok=True)
        sd = {
            "slots": state_dict["slots"].tolist(),
            "used": state_dict["used"].tolist(),
            "last_used": state_dict["last_used"].tolist(),
            "clock": state_dict["clock"],
        }
        with open(self.db_path / "slots.json", "w") as f:
            json.dump(sd, f)

    def load_slots(self) -> dict | None:
        p = self.db_path / "slots.json"
        try:
            with open(p) as f:
                sd = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return {
            "slots": np.array(sd["slots"], dtype=np.float32),
            "used": np.array(sd["used"], dtype=bool),
            "last_used": np.array(sd["last_used"], dtype=np.int64),
            "clock": sd["clock"],
        }

    # ── World Graph ───────────────────────────────────────────────────

    def save_world_graph(self, state_dict: dict):
        self.db_path.mkdir(parents=True, exist_ok=True)
        with open(self.db_path / "world_graph.json", "w") as f:
            json.dump(state_dict, f, indent=2)

    def load_world_graph(self) -> dict | None:
        p = self.db_path / "world_graph.json"
        try:
            with open(p) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # ── PMI Graph ─────────────────────────────────────────────────────

    def save_pmi_graph(self, state_dict: dict):
        self.db_path.mkdir(parents=True, exist_ok=True)
        with open(self.db_path / "pmi_graph.json", "w") as f:
            json.dump(state_dict, f)

    def load_pmi_graph(self) -> dict | None:
        p = self.db_path / "pmi_graph.json"
        try:
            with open(p) as f:
                sd = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        sd["ni"] = {int(k): v for k, v in sd["ni"].items()}
        return sd
