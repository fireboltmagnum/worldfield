"""Persistence layer — saves/loads WorldField state to disk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Persistence:
    """Manages saving/loading of slots, graph, and metadata.

    All state is stored under the db_path directory as JSON files.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

    def save_slots(self, state_dict: dict):
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
        if not p.exists():
            return None
        with open(p) as f:
            sd = json.load(f)
        return {
            "slots": np.array(sd["slots"], dtype=np.float32),
            "used": np.array(sd["used"], dtype=bool),
            "last_used": np.array(sd["last_used"], dtype=np.int64),
            "clock": sd["clock"],
        }

    def save_graph(self, state_dict: dict):
        sd = {
            "min_support": state_dict["min_support"],
            "pmi_floor": state_dict["pmi_floor"],
            "N": state_dict["N"],
            "ni": {str(k): v for k, v in state_dict["ni"].items()},
            "nij": state_dict["nij"],
        }
        with open(self.db_path / "graph.json", "w") as f:
            json.dump(sd, f)

    def load_graph(self) -> dict | None:
        p = self.db_path / "graph.json"
        if not p.exists():
            return None
        with open(p) as f:
            sd = json.load(f)
        sd["ni"] = {int(k): v for k, v in sd["ni"].items()}
        return sd

    def save_concepts(self, state_dict: dict):
        with open(self.db_path / "concepts.json", "w") as f:
            json.dump(state_dict, f)

    def load_concepts(self) -> dict | None:
        p = self.db_path / "concepts.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def save_meta(self, meta: dict):
        with open(self.db_path / "meta.json", "w") as f:
            json.dump(meta, f)

    def load_meta(self) -> dict | None:
        p = self.db_path / "meta.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)
