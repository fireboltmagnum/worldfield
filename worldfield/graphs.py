"""Graph helpers that do not depend on NumPy or PyTorch."""

from __future__ import annotations

from itertools import combinations


def edge(i: int, j: int) -> tuple[int, int]:
    """Return a canonical undirected edge tuple."""
    return (min(i, j), max(i, j))


def undirected_edges(n_nodes: int) -> set[tuple[int, int]]:
    """Return all undirected edges for ``n_nodes`` labeled 0..n-1."""
    return {edge(i, j) for i, j in combinations(range(n_nodes), 2)}
