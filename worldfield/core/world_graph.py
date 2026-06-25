"""World Graph — the single source of truth for WorldField.

Concepts are nodes. Relations are edges. Every observation records evidence,
timestamp, modality, and source. ChromaDB is used for search only.
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


@dataclass
class ConceptNode:
    id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    vector: np.ndarray | None = None
    activation_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    examples: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationEdge:
    source_id: str
    predicate: str
    target_id: str
    confidence: float = 0.0
    support_count: int = 0
    sources: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_confirmed: float = 0.0
    modality: str = ""
    source: str = ""
    polarity: bool = True
    start_time: float | None = None
    end_time: float | None = None
    event_time: float = 0.0
    sequence_id: str = ""
    observation_id: str = ""


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


SIMILARITY_THRESHOLD = 0.85


class WorldGraph:
    """Graph database: concepts as nodes, relations as edges.

    Source of truth. ChromaDB is rebuilt from this for search.
    """

    def __init__(self):
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: list[RelationEdge] = []
        self.adjacency: dict[str, dict[str, list[int]]] = {}
        self._surface_index: dict[str, str] = {}
        self._first_letter_index: dict[str, set[str]] = {}

    def _add_alias(self, key: str, cid: str):
        self._surface_index[key] = cid
        self._first_letter_index.setdefault(key[0], set()).add(key)

    # ── Concept operations ────────────────────────────────────────────

    def resolve(self, surface_form: str, vector: np.ndarray | None = None,
                modality: str = "", source: str = "",
                example: str = "",
                skip_levenshtein: bool = False) -> ConceptNode:
        """Resolve a surface form to a concept. Creates or matches.

        Strategy (in order):
        1. Exact alias match
        2. Levenshtein distance ≤ 2 (plurals, tense variants)
        3. Vector similarity > SIMILARITY_THRESHOLD
        """
        key = surface_form.lower().strip()
        if key in self._surface_index:
            cid = self._surface_index[key]
            return self._touch(cid, example)

        # 2. Levenshtein-based alias expansion (same first letter + ≤ 2 edits + prefix overlap)
        if not skip_levenshtein:
            for alias_key in self._first_letter_index.get(key[0], set()):
                cid = self._surface_index[alias_key]
                if (_levenshtein(key, alias_key) <= 2
                        and (key.startswith(alias_key) or alias_key.startswith(key))):
                    self._add_alias(key, cid)
                    self.nodes[cid].aliases.append(surface_form)
                    return self._touch(cid, example)

        # 3. Vector similarity (high threshold)
        if vector is not None:
            match = self._find_similar(vector)
            if match is not None:
                self._add_alias(key, match.id)
                self.nodes[match.id].aliases.append(surface_form)
                return self._touch(match.id, example)

        cid = str(uuid.uuid4())
        now = time.time()
        node = ConceptNode(
            id=cid,
            canonical_name=surface_form,
            aliases=[surface_form],
            vector=vector.copy() if vector is not None else None,
            first_seen=now,
            last_seen=now,
            activation_count=1,
            examples=[example] if example else [],
            confidence=0.1,
            metadata={
                "modality": modality,
                "source": source,
                "created_at": now,
            },
        )
        self.nodes[cid] = node
        self._add_alias(key, cid)
        return node

    def _find_similar(self, vector: np.ndarray) -> ConceptNode | None:
        """Find existing concept whose vector is similar."""
        best = None
        best_sim = SIMILARITY_THRESHOLD
        for node in self.nodes.values():
            if node.vector is not None and node.vector.shape == vector.shape:
                sim = cosine_sim(vector, node.vector)
                if sim > best_sim:
                    best_sim = sim
                    best = node
        return best

    def _touch(self, cid: str, example: str = "") -> ConceptNode:
        """Update activation without creating a new concept."""
        now = time.time()
        node = self.nodes[cid]
        node.last_seen = now
        node.activation_count += 1
        node.confidence = 1.0 - 1.0 / (1.0 + node.activation_count * 0.5)
        if example and example not in node.examples:
            node.examples.append(example)
        return node

    def add_concept(self, name: str, canonical_name: str = "",
                     vector: np.ndarray | None = None,
                     confidence: float = 0.5):
        """Add a concept node directly (for testing).

        If no vector is provided, a default 16-dim hash-based embedding
        is created from the name so that the node is usable for similarity.
        """
        key = name.lower().strip()
        if key in self._surface_index:
            return self.nodes[self._surface_index[key]]
        if vector is None:
            dim = 16
            vec = np.zeros(dim)
            for ch in name:
                vec[hash(ch) % dim] += 1.0
            norm = np.linalg.norm(vec)
            vector = vec / norm if norm > 0 else vec
        cid = str(uuid.uuid4())
        now = time.time()
        node = ConceptNode(
            id=cid,
            canonical_name=canonical_name or name,
            aliases=[name],
            vector=vector,
            confidence=confidence,
            first_seen=now,
            last_seen=now,
        )
        self.nodes[cid] = node
        self._add_alias(key, cid)
        return node

    def add_relation(self, source: str, predicate: str, target: str,
                     confidence: float = 0.5):
        """Add a relation edge directly (for testing)."""
        src = self.get_concept(source)
        tgt = self.get_concept(target)
        if src is None or tgt is None:
            raise KeyError(f"Concept not found: source={source}, target={target}")
        eid = len(self.edges)
        now = time.time()
        edge = RelationEdge(
            source_id=src.id,
            predicate=predicate,
            target_id=tgt.id,
            confidence=confidence,
            support_count=1,
            first_seen=now,
            last_seen=now,
            last_confirmed=now,
        )
        self.edges.append(edge)
        if src.id not in self.adjacency:
            self.adjacency[src.id] = {}
        if predicate not in self.adjacency[src.id]:
            self.adjacency[src.id][predicate] = []
        self.adjacency[src.id][predicate].append(eid)
        if tgt.id not in self.adjacency:
            self.adjacency[tgt.id] = {}
        rev = f"~{predicate}"
        if rev not in self.adjacency[tgt.id]:
            self.adjacency[tgt.id][rev] = []
        self.adjacency[tgt.id][rev].append(eid)
        return edge

    def get_concept(self, name: str) -> ConceptNode | None:
        key = name.lower().strip()
        cid = self._surface_index.get(key)
        if cid and cid in self.nodes:
            return self.nodes[cid]
        for node in self.nodes.values():
            if node.canonical_name.lower() == key:
                self._add_alias(key, node.id)
                return node
        return None

    def has_concept(self, name: str) -> bool:
        return self.get_concept(name) is not None

    # ── Relation operations ───────────────────────────────────────────

    def relate(self, source_name: str, predicate: str, target_name: str,
               event_time: float | None = None, sequence_id: str = "",
               observation_id: str = "", modality: str = "", source: str = "",
               example: str = "") -> RelationEdge:
        """Add or strengthen a relation between two concepts.

        Creates concepts if they don't exist.
        """
        src = self.get_concept(source_name) or self.resolve(source_name)
        tgt = self.get_concept(target_name) or self.resolve(target_name)
        now = event_time or time.time()

        # Check if this edge already exists
        existing = self._find_edge(src.id, predicate, tgt.id)
        if existing is not None:
            existing.support_count += 1
            existing.last_seen = now
            existing.last_confirmed = now
            if source and source not in existing.sources:
                existing.sources.append(source)
            existing.confidence = 1.0 - 1.0 / (1.0 + existing.support_count * 0.5)
            if example and example not in existing.examples:
                existing.examples.append(example)
            return existing

        eid = len(self.edges)
        edge = RelationEdge(
            source_id=src.id,
            predicate=predicate,
            target_id=tgt.id,
            confidence=0.5,
            support_count=1,
            sources=[source] if source else [],
            examples=[example] if example else [],
            first_seen=now,
            last_seen=now,
            last_confirmed=now,
            event_time=now,
            sequence_id=sequence_id,
            observation_id=observation_id,
            modality=modality,
            source=source,
        )
        self.edges.append(edge)

        # Update adjacency index
        if src.id not in self.adjacency:
            self.adjacency[src.id] = {}
        if predicate not in self.adjacency[src.id]:
            self.adjacency[src.id][predicate] = []
        self.adjacency[src.id][predicate].append(eid)

        # Also index reverse direction for lookup
        if tgt.id not in self.adjacency:
            self.adjacency[tgt.id] = {}
        rev_key = f"~{predicate}"
        if rev_key not in self.adjacency[tgt.id]:
            self.adjacency[tgt.id][rev_key] = []
        self.adjacency[tgt.id][rev_key].append(eid)

        return edge

    def _find_edge(self, source_id: str, predicate: str,
                   target_id: str) -> RelationEdge | None:
        if source_id in self.adjacency and predicate in self.adjacency[source_id]:
            for eid in self.adjacency[source_id][predicate]:
                e = self.edges[eid]
                if e.target_id == target_id:
                    return e
        return None

    def get_relations(self, concept_name: str,
                      predicate: str | None = None) -> list[RelationEdge]:
        """Get outgoing relations for a concept."""
        node = self.get_concept(concept_name)
        if node is None or node.id not in self.adjacency:
            return []
        result = []
        for pred, edges in self.adjacency[node.id].items():
            if predicate is not None and pred != predicate:
                continue
            for eid in edges:
                result.append(self.edges[eid])
        return result

    def get_incoming(self, concept_name: str,
                     predicate: str | None = None) -> list[RelationEdge]:
        """Get incoming relations (reverse)."""
        return self.get_relations(concept_name,
                                  f"~{predicate}" if predicate else None)

    def query(self, concept_name: str, hops: int = 1,
              min_confidence: float = 0.0) -> dict[str, list[dict]]:
        """Simple graph traversal. Returns related concepts and paths."""
        node = self.get_concept(concept_name)
        if node is None:
            return {}

        visited = {node.id}
        frontier = [(node.id, [])]
        results = {}

        for _ in range(hops):
            next_frontier = []
            for cid, path in frontier:
                if cid not in self.adjacency:
                    continue
                for pred, eids in self.adjacency[cid].items():
                    if pred.startswith("~"):
                        continue
                    for eid in eids:
                        e = self.edges[eid]
                        if e.confidence < min_confidence:
                            continue
                        tid = e.target_id
                        if tid in visited:
                            continue
                        visited.add(tid)
                        tgt = self.nodes[tid]
                        entry = {
                            "concept": tgt.canonical_name,
                            "predicate": pred,
                            "confidence": e.confidence,
                            "support": e.support_count,
                            "path": path + [pred],
                        }
                        results.setdefault(tgt.canonical_name, []).append(entry)
                        next_frontier.append((tid, path + [pred]))
            frontier = next_frontier

        return results

    # ── Observation recording ─────────────────────────────────────────

    def record_observation(self, text: str, concepts: list[dict],
                           relations: list[dict],
                           modality: str = "text", source: str = "",
                           event_time: float | None = None,
                           sequence_id: str = "") -> str:
        """Record a full observation: multiple concepts + relations atomically.

        Args:
            text: original input text
            concepts: [{"name": str, "vector": ndarray, ...}]
            relations: [{"source": str, "predicate": str, "target": str, ...}]
            modality: "text", "image", etc.
            source: original source identifier

        Returns: observation_id
        """
        obs_id = str(uuid.uuid4())
        now = event_time or time.time()

        for c in concepts:
            self.resolve(
                surface_form=c["name"],
                vector=c.get("vector"),
                modality=modality,
                source=source,
                example=text,
            )

        for r in relations:
            self.relate(
                source_name=r["source"],
                predicate=r["predicate"],
                target_name=r["target"],
                event_time=now,
                sequence_id=sequence_id,
                observation_id=obs_id,
                modality=modality,
                source=source,
                example=text,
            )

        return obs_id

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def n_concepts(self) -> int:
        return len(self.nodes)

    @property
    def n_relations(self) -> int:
        return len(self.edges)

    @property
    def avg_confidence(self) -> float:
        if not self.nodes:
            return 0.0
        return float(np.mean([n.confidence for n in self.nodes.values()]))

    def top_concepts(self, k: int = 20) -> list[tuple[str, float]]:
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.confidence,
            reverse=True,
        )
        return [(n.canonical_name, n.confidence) for n in sorted_nodes[:k]]

    # ── Serialization ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "canonical_name": n.canonical_name,
                    "aliases": n.aliases,
                    "vector": n.vector.tolist() if n.vector is not None else None,
                    "activation_count": n.activation_count,
                    "first_seen": n.first_seen,
                    "last_seen": n.last_seen,
                    "examples": n.examples,
                    "properties": n.properties,
                    "confidence": n.confidence,
                    "metadata": n.metadata,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": e.source_id,
                    "predicate": e.predicate,
                    "target_id": e.target_id,
                    "confidence": e.confidence,
                    "support_count": e.support_count,
                    "sources": e.sources,
                    "examples": e.examples,
                    "first_seen": e.first_seen,
                    "last_seen": e.last_seen,
                    "last_confirmed": e.last_confirmed,
                    "event_time": e.event_time,
                    "sequence_id": e.sequence_id,
                    "observation_id": e.observation_id,
                    "modality": e.modality,
                    "source": e.source,
                    "polarity": e.polarity,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                }
                for e in self.edges
            ],
        }

    def load_state_dict(self, sd: dict):
        self.nodes = {}
        self._surface_index = {}
        self._first_letter_index = {}
        for nid, data in sd["nodes"].items():
            vec = np.array(data["vector"], dtype=np.float32) if data.get("vector") else None
            self.nodes[nid] = ConceptNode(
                id=data["id"],
                canonical_name=data["canonical_name"],
                aliases=data.get("aliases", []),
                vector=vec,
                activation_count=data.get("activation_count", 0),
                first_seen=data.get("first_seen", 0.0),
                last_seen=data.get("last_seen", 0.0),
                examples=data.get("examples", []),
                properties=data.get("properties", {}),
                confidence=data.get("confidence", 0.0),
                metadata=data.get("metadata", {}),
            )
            self._add_alias(data["canonical_name"].lower(), nid)
            for alias in data.get("aliases", []):
                self._add_alias(alias.lower(), nid)

        self.edges = []
        self.adjacency = {}
        for i, edata in enumerate(sd["edges"]):
            edge = RelationEdge(
                source_id=edata["source_id"],
                predicate=edata["predicate"],
                target_id=edata["target_id"],
                confidence=edata.get("confidence", 0.0),
                support_count=edata.get("support_count", 0),
                sources=edata.get("sources", []),
                examples=edata.get("examples", []),
                first_seen=edata.get("first_seen", 0.0),
                last_seen=edata.get("last_seen", 0.0),
                last_confirmed=edata.get("last_confirmed", 0.0),
                event_time=edata.get("event_time", 0.0),
                sequence_id=edata.get("sequence_id", ""),
                observation_id=edata.get("observation_id", ""),
                modality=edata.get("modality", ""),
                source=edata.get("source", ""),
                polarity=edata.get("polarity", True),
                start_time=edata.get("start_time"),
                end_time=edata.get("end_time"),
            )
            self.edges.append(edge)
            # Rebuild adjacency
            sid = edge.source_id
            tid = edge.target_id
            pred = edge.predicate
            if sid not in self.adjacency:
                self.adjacency[sid] = {}
            if pred not in self.adjacency[sid]:
                self.adjacency[sid][pred] = []
            self.adjacency[sid][pred].append(i)
            if tid not in self.adjacency:
                self.adjacency[tid] = {}
            rev = f"~{pred}"
            if rev not in self.adjacency[tid]:
                self.adjacency[tid][rev] = []
            self.adjacency[tid][rev].append(i)
