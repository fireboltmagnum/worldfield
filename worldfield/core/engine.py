"""Engine — the main orchestrator for WorldField's continuous learning loop.

The engine ties together: encoders → fragment store → slot memory → graph.
It operates in continuous mode (no epochs) — each input is processed immediately.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..device import pick_device
from ..db.store import Persistence
from .fragments import FragmentStore
from .slots import SlotMemory
from .graph import PMIGraph
from .concepts import ConceptMemory


class Engine:
    """The main WorldField engine. Processes inputs in a continuous loop.

    Usage:
        engine = Engine()
        result = engine.process("a red square")       # text
        result = engine.process_image("photo.png")     # image
        result = engine.process_video("clip.mp4")      # video
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.device = self.cfg.device if self.cfg.device != "auto" else pick_device()

        # Lazy-loaded encoders
        self._text_encoder = None
        self._image_encoder = None
        self._video_encoder = None

        # State
        self.store = FragmentStore(self.cfg.db_path)
        self.slots = SlotMemory(
            dim=self.cfg.latent_dim,
            n_slots=self.cfg.n_slots,
            decay=self.cfg.slot_decay,
            merge_threshold=self.cfg.merge_threshold,
        )
        self.graph = PMIGraph(
            min_support=self.cfg.graph_min_support,
            pmi_floor=self.cfg.graph_pmi_floor,
        )
        self.concepts = ConceptMemory()
        self.persistence = Persistence(self.cfg.db_path)

        # Load previous state if available
        self._load_state()

        # Fragment counter for assigning integer IDs to graph nodes
        self._frag_counter = 0
        self._frag_to_node: dict[str, int] = {}

    # ── Encoder accessors (lazy) ──────────────────────────────────────

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            from ..encoders.text import TextEncoder
            self._text_encoder = TextEncoder(
                latent_dim=self.cfg.latent_dim,
                model_name=self.cfg.text_model,
                device=self.device,
            )
        return self._text_encoder

    @property
    def image_encoder(self):
        if self._image_encoder is None:
            from ..encoders.image import load_image_encoder
            # Try loading pretrained checkpoint
            ckpt = Path("day_one/out/worldfield_rich.pt")
            if ckpt.exists():
                self._image_encoder = load_image_encoder(
                    str(ckpt), self.cfg, device=self.device
                )
            else:
                from ..encoders.image import ImageEncoder
                enc = ImageEncoder(self.cfg)
                enc.to(self.device)
                enc.eval()
                self._image_encoder = enc
        return self._image_encoder

    @property
    def video_encoder(self):
        if self._video_encoder is None:
            from ..encoders.video import VideoEncoder
            self._video_encoder = VideoEncoder(
                self.image_encoder, self.cfg, device=self.device
            )
        return self._video_encoder

    # ── Core processing ──────────────────────────────────────────────

    def process(self, text: str) -> dict[str, Any]:
        """Process a text input through the full pipeline."""
        vec = self.text_encoder.encode(text)
        return self._process_vector(vec, {"modality": "text", "raw": text, "timestamp": time.time()})

    def process_image(self, image_path: str) -> dict[str, Any]:
        """Process an image through the full pipeline."""
        from PIL import Image
        import torch
        img = Image.open(image_path).resize((self.cfg.img_size, self.cfg.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        vec = self.image_encoder.encode(tensor)[0]
        return self._process_vector(vec, {"modality": "image", "raw": image_path, "timestamp": time.time()})

    def process_video(self, video_path: str) -> dict[str, Any]:
        """Process a video through the full pipeline."""
        vec = self.video_encoder.encode(video_path)
        return self._process_vector(vec, {"modality": "video", "raw": video_path, "timestamp": time.time()})

    def _process_vector(self, vec: np.ndarray, metadata: dict) -> dict[str, Any]:
        """Shared pipeline: store → slot → graph → retrieve → reason."""
        # 1. Store as fragment
        fid = self.store.add(vec, metadata)
        node_id = self._frag_counter
        self._frag_to_node[fid] = node_id
        self._frag_counter += 1

        # 2. Track concept
        label = metadata.get("raw", str(metadata.get("timestamp", "")))[:64]
        concept_name = self.concepts.observe(fid, vec, label=label)

        # 3. Update slot memory
        slot_state = self.slots.update(vec)

        # 4. Update graph with co-active slot fragments
        self.graph.observe([node_id])

        # 4. Retrieve related fragments
        related = self.store.search(vec, k=self.cfg.retrieval_k)

        # 5. Graph reasoning
        reasoning = {}
        if self.graph.has_edges:
            scores = self.graph.propagate(
                np.array([node_id]),
                hops=self.cfg.refine_hops,
                decay=self.cfg.refine_decay,
            )
            reasoning["n_edges"] = self.graph.n_edges
            reasoning["top_nodes"] = self._top_graph_nodes(scores, 5)

        # 6. Build result
        concept = self.concepts.concepts.get(concept_name)
        result = {
            "fragment_id": fid,
            "concept_name": concept_name,
            "concept_confidence": concept.confidence if concept else 0.0,
            "concept_uncertainty": concept.uncertainty if concept else 1.0,
            "slot_state_active": self.slots.active_count(),
            "slot_state": slot_state.tolist(),
            "related": related,
            "reasoning": reasoning,
            "total_fragments": self.store.count,
        }

        # Persist after each step
        self._save_state()

        return result

    def _top_graph_nodes(self, scores: np.ndarray, k: int) -> list[dict]:
        top_idx = np.argsort(scores)[::-1][:k]
        out = []
        for idx in top_idx:
            if scores[idx] > 0:
                out.append({"node": int(idx), "score": float(scores[idx])})
        return out

    def query(self, text: str) -> dict[str, Any]:
        """Query without storing. Just retrieve + reason."""
        vec = self.text_encoder.encode(text)
        related = self.store.search(vec, k=self.cfg.retrieval_k)
        result = {"query": text, "related": related}

        if self.graph.has_edges:
            # Find nearest fragment nodes and propagate
            if related:
                node_ids = []
                for r in related:
                    fid = r["id"]
                    if fid in self._frag_to_node:
                        node_ids.append(self._frag_to_node[fid])
                if node_ids:
                    scores = self.graph.propagate(
                        np.array(node_ids),
                        hops=self.cfg.refine_hops,
                        decay=self.cfg.refine_decay,
                    )
                    result["reasoning"] = {
                        "n_edges": self.graph.n_edges,
                        "top_nodes": self._top_graph_nodes(scores, 5),
                    }
        return result

    def refine(self, text: str, iters: int | None = None) -> list[dict]:
        """Run iterative refinement on a query. Returns trajectory."""
        iters = iters or self.cfg.refine_iters
        vec = self.text_encoder.encode(text)
        related = self.store.search(vec, k=self.cfg.retrieval_k)
        trajectory = []

        if not related or not self.graph.has_edges:
            return trajectory

        node_ids = []
        for r in related:
            fid = r["id"]
            if fid in self._frag_to_node:
                node_ids.append(self._frag_to_node[fid])

        if not node_ids:
            return trajectory

        state = np.zeros(self._frag_counter, dtype=np.float32)
        state[node_ids] = 1.0
        state /= state.max() + 1e-12

        for i in range(iters):
            scores = self.graph.propagate(
                np.where(state > 0)[0],
                state[state > 0],
                hops=self.cfg.refine_hops,
                decay=self.cfg.refine_decay,
            )
            top_k = int(len(scores) * self.cfg.refine_keep_frac)
            threshold = np.sort(scores)[::-1][min(top_k, len(scores) - 1)]
            new_state = np.where(scores >= threshold, scores, 0.0)
            new_state /= new_state.max() + 1e-12
            state = self.cfg.refine_damping * state + (1 - self.cfg.refine_damping) * new_state
            state[state < 1e-4] = 0.0
            trajectory.append({
                "iter": i,
                "active_nodes": int(np.sum(state > 0)),
                "top_node": int(np.argmax(state)) if state.sum() > 0 else -1,
                "top_score": float(np.max(state)),
            })

        return trajectory

    # ── Persistence ──────────────────────────────────────────────────

    def _save_state(self):
        self.persistence.save_slots(self.slots.state_dict())
        self.persistence.save_graph(self.graph.state_dict())
        self.persistence.save_concepts(self.concepts.state_dict())
        self.persistence.save_meta({
            "frag_counter": self._frag_counter,
            "frag_to_node": self._frag_to_node,
        })

    def _load_state(self):
        slots_sd = self.persistence.load_slots()
        if slots_sd:
            self.slots.load_state_dict(slots_sd)
        graph_sd = self.persistence.load_graph()
        if graph_sd:
            self.graph.load_state_dict(graph_sd)
        concepts_sd = self.persistence.load_concepts()
        if concepts_sd:
            self.concepts.load_state_dict(concepts_sd)
        meta = self.persistence.load_meta()
        if meta:
            self._frag_counter = meta.get("frag_counter", 0)
            self._frag_to_node = meta.get("frag_to_node", {})

    def reset(self):
        """Reset all state (keeps DB)."""
        self.slots.reset()
        self.graph = PMIGraph(
            min_support=self.cfg.graph_min_support,
            pmi_floor=self.cfg.graph_pmi_floor,
        )
        self._frag_counter = 0
        self._frag_to_node = {}
        self._save_state()
