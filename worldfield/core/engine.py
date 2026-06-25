"""Engine — orchestrates the full WorldField pipeline.

Now uses the World Graph as source of truth + NLP pipeline for concept extraction.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..device import pick_device
from ..db.store import Persistence
from .slots import SlotMemory
from .graph import PMIGraph
from .world_graph import WorldGraph
from .activation import ActivationEngine
from .world_state import WorldStateBuilder
from ..reasoning.inference import InferenceEngine

class Engine:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.device = self.cfg.device if self.cfg.device != "auto" else pick_device()

        self._text_encoder = None
        self._image_encoder = None
        self._video_encoder = None
        self._nlp = None
        self._extractor = None
        self._resolver = None

        # Primary source of truth
        self.graph = WorldGraph()

        # Working memory (dimension determined by the text encoder)
        self.slots = None

        # Activation layer (spreading activation over the concept graph)
        self.activator = ActivationEngine(
            graph=self.graph,
            decay_rate=self.cfg.activation_decay,
            spread_factor=self.cfg.activation_spread,
            spread_hops=self.cfg.activation_hops,
        )

        # World state builder (current reality model)
        self.world_builder = WorldStateBuilder(graph=self.graph)

        # Inference engine (reasoning over world state)
        self.inference_engine = InferenceEngine(
            graph=self.graph,
            max_inheritance_depth=self.cfg.inference_depth,
        )

        # Secondary association layer
        self.pmi = PMIGraph(
            min_support=self.cfg.graph_min_support,
            pmi_floor=self.cfg.graph_pmi_floor,
        )

        # Search index (built from World Graph)
        self._search_ready = False

        self.persistence = Persistence(self.cfg.db_path)
        try:
            self._load_state()
        except FileNotFoundError:
            pass

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            from ..encoders.text import TextEncoder
            self._text_encoder = TextEncoder(
                latent_dim=self.cfg.latent_dim,
                model_name=self.cfg.text_model,
                device=self.device,
                use_projection=False,
            )
            if self.slots is None:
                from .slots import SlotMemory
                self.slots = SlotMemory(
                    dim=self._text_encoder.dim,
                    n_slots=self.cfg.n_slots,
                    decay=self.cfg.slot_decay,
                    merge_threshold=self.cfg.merge_threshold,
                )
        return self._text_encoder

    @property
    def image_encoder(self):
        if self._image_encoder is None:
            from ..encoders.image import load_image_encoder
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

    @property
    def nlp(self):
        if self._nlp is None:
            from ..nlp.parser import NLPParser
            self._nlp = NLPParser()
        return self._nlp

    @property
    def extractor(self):
        if self._extractor is None:
            from ..nlp.extractor import ConceptExtractor
            self._extractor = ConceptExtractor(self.nlp)
        return self._extractor

    @property
    def resolver(self):
        if self._resolver is None:
            from ..nlp.resolver import ConceptResolver
            self._resolver = ConceptResolver(self.graph)
        return self._resolver

    # ── Core processing ──────────────────────────────────────────────

    def process(self, text: str) -> dict[str, Any]:
        """NLP → extract → resolve → graph → slots → search."""
        timings = {}
        t0 = time.perf_counter()

        # 1. NLP: extract concepts and relations
        concepts, relations = self.extractor.extract_with_vectors(
            text, self.text_encoder, self.cfg
        )
        timings["understanding"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()

        # 2. Resolve surface forms to canonical concepts
        resolved_conc, resolved_rel = self.resolver.extract_and_resolve(
            text, concepts, relations, modality="text", source="user_input"
        )
        timings["resolve"] = (time.perf_counter() - t1) * 1000
        t_act = time.perf_counter()

        # 3. Activate concepts and spread to related
        concept_names = [c["name"] for c in resolved_conc]
        self.activator.trigger(concept_names)
        self.activator.spread()
        active_concepts = self.activator.get_active(threshold=0.05)
        working_set = self.activator.get_working_set(k=10)
        timings["activation"] = (time.perf_counter() - t_act) * 1000
        t_ws = time.perf_counter()

        # 4. Build world state (current reality model)
        world_state = self.world_builder.from_activations(
            active_concepts, resolved_rel
        )
        timings["world_state"] = (time.perf_counter() - t_ws) * 1000
        t_inf = time.perf_counter()

        # 5. Run inference over the world state
        inference_result = self.inference_engine.reason(world_state)
        timings["reasoning"] = (time.perf_counter() - t_inf) * 1000
        t2 = time.perf_counter()

        # 6. Record graph state before update
        pre_concepts = self.graph.n_concepts
        pre_relations = self.graph.n_relations

        obs_id = self.graph.record_observation(
            text=text,
            concepts=resolved_conc,
            relations=resolved_rel,
            modality="text",
            source="user_input",
        )
        timings["world_update"] = (time.perf_counter() - t2) * 1000
        t3 = time.perf_counter()

        # 4. Update slot memory with concept vectors
        slot_updates = []
        if self.slots is not None:
            for c in resolved_conc:
                if c.get("vector") is not None:
                    self.slots.update(c["vector"])
                    slot_updates.append(c["name"])
        timings["slot_update"] = (time.perf_counter() - t3) * 1000
        t4 = time.perf_counter()

        # 5. Update PMI graph (association layer) — uses compact sequential IDs
        if not hasattr(self, "_pmi_ids"):
            self._pmi_ids = {}
            self._pmi_next = 0
        concept_ids = []
        for c in resolved_conc:
            name = c["name"]
            if name not in self._pmi_ids:
                self._pmi_ids[name] = self._pmi_next
                self._pmi_next += 1
            concept_ids.append(self._pmi_ids[name])
        self.pmi.observe(concept_ids)
        timings["pmi"] = (time.perf_counter() - t4) * 1000
        t5 = time.perf_counter()

        # 6. Graph reasoning
        query_concept = resolved_conc[0]["name"] if resolved_conc else ""
        related_concepts = {}
        if query_concept and self.graph.n_concepts > 0:
            related_concepts = self.graph.query(query_concept, hops=1)
        timings["graph_query"] = (time.perf_counter() - t5) * 1000

        # 7. Build result
        result = {
            "observation_id": obs_id,
            "text": text,
            "concepts_extracted": [c["name"] for c in resolved_conc],
            "extracted_concepts_raw": resolved_conc,
            "extracted_relations_raw": resolved_rel,
            "relations_extracted": [
                f"{r['source']} -[{r['predicate']}]-> {r['target']}"
                for r in resolved_rel
            ],
            "graph_pre_state": (pre_concepts, pre_relations),
            "slot_concepts": slot_updates,
            "slot_state_active": self.slots.active_count() if self.slots else 0,
            "activation_active": active_concepts,
            "activation_working_set": working_set,
            "world_state": world_state.to_dict(),
            "inference_result": inference_result.to_dict(),
            "graph_query": related_concepts,
            "total_concepts": self.graph.n_concepts,
            "total_relations": self.graph.n_relations,
            "timings": timings,
        }

        # Decay activation for next turn
        self.activator.tick()

        self._save_state()
        return result

    def process_image(self, image, source: str = "") -> dict[str, Any]:
        """Process an image through the vision pipeline."""
        import time
        timings = {}
        t0 = time.perf_counter()

        # Encode image
        vec = self.image_encoder.encode(image)
        timings["image_encoding"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()

        # Create a concept from the image vector
        name = f"image:{source}" if source else "image:unknown"
        concept = {"name": name, "vector": vec, "pos": "IMAGE", "is_entity": True}
        timings["concept_formation"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()

        pre_concepts = self.graph.n_concepts
        pre_relations = self.graph.n_relations

        self.activator.trigger([name])
        self.activator.spread()
        active_img = self.activator.get_active(threshold=0.05)
        timings["activation"] = (time.perf_counter() - t2) * 1000
        t3 = time.perf_counter()

        ws_img = self.world_builder.from_activations(active_img, [])
        timings["world_state"] = (time.perf_counter() - t3) * 1000
        t_inf = time.perf_counter()

        inference_result = self.inference_engine.reason(ws_img)
        timings["reasoning"] = (time.perf_counter() - t_inf) * 1000

        obs_id = self.graph.record_observation(
            text=f"[image] {source}",
            concepts=[concept],
            relations=[],
            modality="image",
            source=source,
        )
        timings["world_update"] = (time.perf_counter() - t2) * 1000

        result = {
            "observation_id": obs_id,
            "text": f"[image] {source}",
            "concepts_extracted": [name],
            "extracted_concepts_raw": [concept],
            "extracted_relations_raw": [],
            "relations_extracted": [],
            "graph_pre_state": (pre_concepts, pre_relations),
            "graph_query": self.graph.query(name, hops=1) if self.graph.n_concepts > 0 else {},
            "activation_active": active_img,
            "activation_working_set": self.activator.get_working_set(k=10),
            "world_state": ws_img.to_dict(),
            "inference_result": inference_result.to_dict(),
            "slot_concepts": [name],
            "slot_state_active": self.slots.active_count() if self.slots else 0,
            "total_concepts": self.graph.n_concepts,
            "total_relations": self.graph.n_relations,
            "timings": timings,
        }
        self.activator.tick()
        self._save_state()
        return result

    def query(self, text: str, hops: int = 2) -> dict[str, Any]:
        """Query: extract concepts → traverse graph."""
        concepts, _ = self.extractor.extract_with_vectors(text, self.text_encoder)
        query_name = concepts[0]["name"] if concepts else text.lower().strip()

        node = self.graph.get_concept(query_name)
        if node is None:
            # Search by vector similarity against all concept vectors
            vec = self.text_encoder.encode(query_name)
            best_sim = -1
            best_name = query_name
            for n in self.graph.nodes.values():
                if n.vector is not None:
                    sim = float(np.dot(vec, n.vector) /
                                (np.linalg.norm(vec) * np.linalg.norm(n.vector) + 1e-12))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = n.canonical_name
            query_name = best_name if best_sim > 0.5 else query_name

        related = self.graph.query(query_name, hops=hops)
        flat = []
        for concept, entries in related.items():
            for e in entries:
                flat.append({
                    "concept": concept,
                    "relation": e["predicate"],
                    "via": query_name,
                    "confidence": e["confidence"],
                    "support": e["support"],
                })

        return {
            "query": text,
            "resolved_concept": query_name,
            "related_concepts": flat,
            "total_traversed": len(flat),
        }

    def refine(self, text: str, hops: int = 2) -> list[dict]:
        """Iterative graph traversal for a query."""
        result = self.query(text, hops=hops)
        return [result]

    # ── Persistence ──────────────────────────────────────────────────

    def _save_state(self):
        if self.slots is not None:
            self.persistence.save_slots(self.slots.state_dict())
        self.persistence.save_world_graph(self.graph.state_dict())
        self.persistence.save_activation(self.activator.state_dict())
        extra = {
            "pmi_ids": getattr(self, "_pmi_ids", {}),
            "pmi_next": getattr(self, "_pmi_next", 0),
        }
        self.persistence.save_pmi_graph({**self.pmi.state_dict(), **extra})

    def _load_state(self):
        slots_sd = self.persistence.load_slots()
        if slots_sd and self.slots is not None:
            self.slots.load_state_dict(slots_sd)
        graph_sd = self.persistence.load_world_graph()
        if graph_sd:
            self.graph.load_state_dict(graph_sd)
        act_sd = self.persistence.load_activation()
        if act_sd:
            self.activator.load_state_dict(act_sd)
        pmi_sd = self.persistence.load_pmi_graph()
        if pmi_sd:
            self.pmi.load_state_dict(pmi_sd)
            self._pmi_ids = pmi_sd.get("pmi_ids", {})
            self._pmi_next = pmi_sd.get("pmi_next", 0)

    def stats(self) -> dict[str, Any]:
        return {
            "concepts": self.graph.n_concepts,
            "relations": self.graph.n_relations,
            "observations": 0,
            "slots_used": self.slots.active_count() if self.slots is not None else 0,
            "avg_confidence": self.graph.avg_confidence,
        }

    def reset(self):
        if self.slots is not None:
            self.slots.reset()
        self.graph = WorldGraph()
        self.activator = ActivationEngine(graph=self.graph)
        self.pmi = PMIGraph(
            min_support=self.cfg.graph_min_support,
            pmi_floor=self.cfg.graph_pmi_floor,
        )
        self._save_state()
