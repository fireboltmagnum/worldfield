from __future__ import annotations

from collections import deque
from typing import Any

from ..core.world_graph import WorldGraph


class GraphOps:
    def __init__(self, graph: WorldGraph):
        self.graph = graph

    def neighbors(self, concept: str,
                  predicate: str | None = None,
                  direction: str = "outgoing",
                  min_confidence: float = 0.0) -> list[dict[str, Any]]:
        node = self.graph.get_concept(concept)
        if node is None:
            return []
        results = []
        if direction in ("outgoing", "both"):
            for edge in self.graph.get_relations(concept, predicate):
                if edge.confidence < min_confidence:
                    continue
                tgt = self.graph.nodes.get(edge.target_id)
                if tgt is None:
                    continue
                results.append({
                    "concept": tgt.canonical_name,
                    "predicate": edge.predicate,
                    "confidence": edge.confidence,
                    "count": edge.support_count,
                    "sources": edge.sources,
                    "polarity": edge.polarity,
                    "direction": "outgoing",
                })
        if direction in ("incoming", "both"):
            for edge in self.graph.get_incoming(concept, predicate):
                if edge.confidence < min_confidence:
                    continue
                src = self.graph.nodes.get(edge.source_id)
                if src is None:
                    continue
                results.append({
                    "concept": src.canonical_name,
                    "predicate": edge.predicate,
                    "confidence": edge.confidence,
                    "count": edge.support_count,
                    "sources": edge.sources,
                    "polarity": edge.polarity,
                    "direction": "incoming",
                })
        return results

    def transitive_closure(self, concept: str,
                           predicate: str,
                           max_hops: int = 5) -> set[str]:
        node = self.graph.get_concept(concept)
        if node is None:
            return set()
        visited: set[str] = set()
        frontier = {node.id}
        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for cid in frontier:
                if cid in visited:
                    continue
                visited.add(cid)
                for edge in self.graph.edges:
                    if (edge.source_id == cid
                            and edge.predicate == predicate
                            and edge.target_id not in visited):
                        next_frontier.add(edge.target_id)
            frontier = next_frontier
        return {self.graph.nodes[cid].canonical_name
                for cid in visited if cid != node.id}

    def expand_concept(self, concept: str,
                       depth: int = 2,
                       predicates: list[str] | None = None,
                       min_confidence: float = 0.0) -> dict[str, Any]:
        node = self.graph.get_concept(concept)
        if node is None:
            return {"concept": concept, "levels": {}, "total_concepts": 0, "max_depth": 0}
        levels: dict[int, list[dict[str, Any]]] = {}
        visited_ids: set[str] = {node.id}
        frontier = [(node.id, [])]
        for level in range(1, depth + 1):
            next_frontier = []
            level_results: list[dict[str, Any]] = []
            for cid, path in frontier:
                for edge in self.graph.edges:
                    is_outgoing = edge.source_id == cid and edge.target_id not in visited_ids
                    is_incoming = edge.target_id == cid and edge.source_id not in visited_ids
                    if not (is_outgoing or is_incoming):
                        continue
                    if predicates and edge.predicate not in predicates:
                        continue
                    if edge.confidence < min_confidence:
                        continue
                    neighbor_id = edge.target_id if is_outgoing else edge.source_id
                    visited_ids.add(neighbor_id)
                    neighbor = self.graph.nodes.get(neighbor_id)
                    if neighbor is None:
                        continue
                    entry = {
                        "concept": neighbor.canonical_name,
                        "predicate": edge.predicate,
                        "confidence": edge.confidence,
                        "count": edge.support_count,
                        "path": path + [edge.predicate],
                        "direction": "outgoing" if is_outgoing else "incoming",
                        "polarity": edge.polarity,
                    }
                    level_results.append(entry)
                    next_frontier.append((neighbor_id, entry["path"]))
            if level_results:
                levels[level] = level_results
            frontier = next_frontier
        total = len({c["concept"] for lv in levels.values() for c in lv})
        return {"concept": concept, "levels": levels, "total_concepts": total, "max_depth": depth}

    def find_path(self, source: str, target: str,
                  max_hops: int = 5,
                  predicates: list[str] | None = None,
                  min_confidence: float = 0.0) -> list[list[dict[str, Any]]]:
        src_node = self.graph.get_concept(source)
        tgt_node = self.graph.get_concept(target)
        if src_node is None or tgt_node is None:
            return []
        all_paths: list[list[dict[str, Any]]] = []
        visited: set[str] = set()

        def _dfs(current_id: str, target_id: str, path: list[dict[str, Any]], depth: int):
            if depth > max_hops:
                return
            if current_id == target_id and path:
                all_paths.append(list(path))
                return
            visited.add(current_id)
            for edge in self.graph.edges:
                if edge.source_id != current_id:
                    continue
                if predicates and edge.predicate not in predicates:
                    continue
                if edge.confidence < min_confidence:
                    continue
                if edge.target_id in visited:
                    continue
                tgt = self.graph.nodes.get(edge.target_id)
                if tgt is None:
                    continue
                path.append({
                    "source": self.graph.nodes[current_id].canonical_name,
                    "predicate": edge.predicate,
                    "target": tgt.canonical_name,
                    "confidence": edge.confidence,
                    "count": edge.support_count,
                })
                _dfs(edge.target_id, target_id, path, depth + 1)
                path.pop()
            visited.discard(current_id)

        _dfs(src_node.id, tgt_node.id, [], 0)
        return all_paths

    def shortest_path(self, source: str, target: str,
                      max_hops: int = 5) -> list[dict[str, Any]] | None:
        src_node = self.graph.get_concept(source)
        tgt_node = self.graph.get_concept(target)
        if src_node is None or tgt_node is None:
            return None
        queue: deque[tuple[str, list[dict[str, Any]]]] = deque()
        queue.append((src_node.id, []))
        visited: set[str] = {src_node.id}
        while queue:
            current_id, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for edge in self.graph.edges:
                if edge.source_id != current_id:
                    continue
                if edge.target_id in visited:
                    continue
                tgt = self.graph.nodes.get(edge.target_id)
                if tgt is None:
                    continue
                hop = {
                    "source": self.graph.nodes[current_id].canonical_name,
                    "predicate": edge.predicate,
                    "target": tgt.canonical_name,
                    "confidence": edge.confidence,
                    "count": edge.support_count,
                }
                new_path = path + [hop]
                if edge.target_id == tgt_node.id:
                    return new_path
                visited.add(edge.target_id)
                queue.append((edge.target_id, new_path))
        return None

    def find_common_ancestor(self, concepts: list[str],
                             predicate: str = "is_a",
                             max_hops: int = 5) -> list[dict[str, Any]]:
        if not concepts:
            return []
        lineage: list[set[str]] = []
        for c in concepts:
            ancestors = self.transitive_closure(c, predicate, max_hops)
            lineage.append(ancestors | {c})
        if not lineage:
            return []
        common = set.intersection(*lineage)
        if not common:
            return []
        results = []
        for ancestor in common:
            paths = []
            for c in concepts:
                p = self.find_path(c, ancestor, max_hops, [predicate])
                if p:
                    paths.extend(p)
            results.append({"ancestor": ancestor, "paths": paths})
        return results

    def similar_concepts(self, concept: str,
                         k: int = 10,
                         min_overlap: float = 0.3) -> list[tuple[str, float]]:
        node = self.graph.get_concept(concept)
        if node is None:
            return []
        my_edges: set[tuple[str, str, str]] = set()
        for edge in self.graph.edges:
            if edge.source_id == node.id:
                tgt = self.graph.nodes.get(edge.target_id)
                if tgt:
                    my_edges.add((edge.predicate, tgt.canonical_name, "out"))
            if edge.target_id == node.id:
                src = self.graph.nodes.get(edge.source_id)
                if src:
                    my_edges.add((edge.predicate, src.canonical_name, "in"))
        if not my_edges:
            return []
        scores: list[tuple[str, float]] = []
        for other_id, other_node in self.graph.nodes.items():
            if other_id == node.id:
                continue
            other_edges: set[tuple[str, str, str]] = set()
            for edge in self.graph.edges:
                if edge.source_id == other_id:
                    tgt = self.graph.nodes.get(edge.target_id)
                    if tgt:
                        other_edges.add((edge.predicate, tgt.canonical_name, "out"))
                if edge.target_id == other_id:
                    src = self.graph.nodes.get(edge.source_id)
                    if src:
                        other_edges.add((edge.predicate, src.canonical_name, "in"))
            if not other_edges:
                continue
            intersection = my_edges & other_edges
            union = my_edges | other_edges
            overlap = len(intersection) / len(union) if union else 0
            if overlap >= min_overlap:
                scores.append((other_node.canonical_name, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
