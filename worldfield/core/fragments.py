"""Fragment store — persistent vector storage backed by ChromaDB.

Fragments are the atomic units of experience in WorldField. Each fragment is a
latent vector with associated metadata (modality, original input, timestamp).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np


class FragmentStore:
    """Persistent vector store for latent fragments, wrapping ChromaDB.

    Usage:
        store = FragmentStore(db_path="./worldfield_db")
        fid = store.add(vec, {"text": "a red square"})
        results = store.search(vec, k=5)
    """

    def __init__(self, db_path: str, collection_name: str = "fragments"):
        import chromadb
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            count = self._collection.count()
            if count > 0:
                sample = self._collection.get(limit=1, include=["embeddings"])
                if sample["embeddings"]:
                    self._dim = len(sample["embeddings"][0])
            if self._dim is None:
                self._dim = 128
        return self._dim

    @property
    def count(self) -> int:
        return self._collection.count()

    def add(self, vector: np.ndarray, metadata: dict | None = None,
            fragment_id: str | None = None) -> str:
        """Add a single fragment. Returns its ID."""
        fid = fragment_id or str(uuid.uuid4())
        self._collection.add(
            embeddings=[vector.tolist()],
            metadatas=[metadata or {}],
            ids=[fid],
        )
        if self._dim is None:
            self._dim = len(vector)
        return fid

    def add_batch(self, vectors: np.ndarray,
                  metadatas: list[dict] | None = None,
                  ids: list[str] | None = None) -> list[str]:
        """Add multiple fragments at once."""
        n = len(vectors)
        fids = ids or [str(uuid.uuid4()) for _ in range(n)]
        self._collection.add(
            embeddings=vectors.tolist(),
            metadatas=metadatas or [{} for _ in range(n)],
            ids=fids,
        )
        if self._dim is None:
            self._dim = vectors.shape[1]
        return fids

    def search(self, query: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
        """Search for nearest fragments. Returns list of {id, metadata, score}."""
        results = self._collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(k, max(self._collection.count(), 1)),
            include=["metadatas", "distances"],
        )
        out = []
        if results["ids"] and results["ids"][0]:
            for fid, meta, dist in zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                out.append({
                    "id": fid,
                    "metadata": meta,
                    "score": 1.0 - float(dist),
                })
        return out

    def get(self, fragment_id: str) -> dict | None:
        """Retrieve a fragment by ID."""
        results = self._collection.get(
            ids=[fragment_id],
            include=["metadatas", "embeddings"],
        )
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
                "vector": np.array(results["embeddings"][0], dtype=np.float32),
            }
        return None

    def update_metadata(self, fragment_id: str, metadata: dict):
        """Update metadata for an existing fragment."""
        self._collection.update(ids=[fragment_id], metadatas=[metadata])

    def delete(self, fragment_id: str):
        self._collection.delete(ids=[fragment_id])

    def all_fragments(self, limit: int = 1000) -> list[dict]:
        """Return all fragments (with embeddings for graph building)."""
        results = self._collection.get(
            limit=limit,
            include=["metadatas", "embeddings"],
        )
        out = []
        for fid, meta, emb in zip(
            results["ids"], results["metadatas"], results["embeddings"]
        ):
            out.append({
                "id": fid,
                "metadata": meta,
                "vector": np.array(emb, dtype=np.float32),
            })
        return out

    def clear(self):
        """Delete all fragments."""
        all_ids = self._collection.get(limit=self._collection.count())["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
