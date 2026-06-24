# Reasoning Engine — Design Spec

**Date:** 2026-06-24
**Status:** Draft

## Overview

The Reasoning Engine transforms WorldField from a knowledge graph into an
inference system. It answers natural language questions using graph traversal,
inheritance inference, and path finding — returning probabilistic results with
full evidence chains.

## Architecture

```
User Question (string)
    │
    ▼
QueryParser (spaCy + pattern matching)
    │  → StructuredQuery { intent, subject, predicate, object, question_word }
    ▼
ReasoningEngine
    │  → calls GraphOps for all graph access
    │  → intent handlers produce Answer dicts
    │  → inheritance inference creates new conclusions
    ▼
Answer { results: [...], evidence: [...], confidence: float }
```

Four files in `worldfield/reasoning/`:

| File | Responsibility |
|---|---|
| `parser.py` | NL → `StructuredQuery` |
| `graph_ops.py` | Pure graph traversal primitives |
| `engine.py` | Intent handlers + inference |
| `formatter.py` | Answer → readable NL |

## 1. RelationEdge Enhancement

The existing `RelationEdge` in `world_graph.py` gets expanded fields:

```python
@dataclass
class RelationEdge:
    source_id: str            # concept node ID (subject)
    predicate: str            # relation type
    target_id: str            # concept node ID (object)
    confidence: float         # 0.0–1.0
    support_count: int        # how many observations
    sources: list[str]        # provenance, e.g., ["gqa/12345", "coco", "user_input"]
    examples: list[str]       # textual evidence
    first_seen: float         # timestamp of first observation
    last_seen: float          # timestamp of most recent observation
    last_confirmed: float     # timestamp of last independent confirmation
    modality: str             # "text", "image", etc.
    polarity: bool = True     # True = fact holds, False = fact is negated
    start_time: float | None = None   # temporal scope (reserved)
    end_time: float | None = None     # temporal scope (reserved)
```

Key additions:

- **`polarity`**: Stores contradictory evidence. `bird -[can]-> fly (polarity=True)`
  alongside `penguin -[can]-> fly (polarity=False)`. Both coexist in the graph;
  inheritance resolves the contradiction by specificity (child overrides parent).

- **`last_confirmed`**: Enables staleness detection for continuous learning.
  A relation not seen recently can have its confidence decayed.

- **`start_time` / `end_time`**: Reserved for temporal reasoning. Enables future
  queries like "Where did the cat sleep before noon?" or "Was the cat on the
  table yesterday?". Not implemented in V1.

## 2. StructuredQuery

```python
@dataclass
class StructuredQuery:
    intent: str              # see Intent Types below
    subject: str | None      # primary concept
    predicate: str | None    # relation (if applicable)
    object: str | None       # target concept (if applicable)
    question_word: str       # "what", "where", "is", "how", etc.
    raw_text: str            # original question
```

### Intent Types (architecture-ready, partial implementation)

| Intent | Example | Priority |
|---|---|---|---|
| `FACT_LOOKUP` | "What is a cat?" | Implement |
| `RELATION_QUERY` | "What does a cat sit on?" | Implement |
| `REVERSE_QUERY` | "What sits on a table?" | Implement |
| `HIERARCHY_CHECK` | "Is a cat an animal?" | Implement |
| `PATH_FINDING` | "How is cat related to dog?" | Implement |
| `COMPARISON` | "How are cats and dogs different?" | Scaffold |
| `MISSING_LINK` | "What properties might cat and dog share?" | Scaffold |
| `ANALOGY` | "What is to cat as bone is to dog?" | Scaffold |
| `CAUSAL_QUERY` | "What causes rain?" | Scaffold |
| `TEMPORAL_QUERY` | "What happens after sunset?" | Scaffold |
| `HYPOTHESIS` | "What if all cats were dogs?" | Scaffold |

**MISSING_LINK** is the most important future intent. Given two concepts with
similar profiles (shared relations to shared targets), MISSING_LINK predicts
likely but unobserved relations. E.g., cat→mammal and dog→mammal → the engine
asks "cat and dog share X relations; what relations does one have that the
other is missing?" This is where WorldField starts generating knowledge
instead of retrieving it.

## 3. GraphOps

Pure functions over `WorldGraph`. ReasoningEngine never touches the graph
directly — it calls GraphOps.

```python
class GraphOps:
    def __init__(self, graph: WorldGraph): ...

    def neighbors(self, concept: str,
                  predicate: str | None = None,
                  direction: str = "outgoing",
                  min_confidence: float = 0.0) -> list[dict]:
        """Get neighboring concepts, optionally filtered.
        direction: "outgoing", "incoming", or "both"
        Returns: [{"concept": str, "predicate": str, "confidence": float,
                    "count": int, "edge_id": int}, ...]
        """

    def find_path(self, source: str, target: str,
                  max_hops: int = 5,
                  predicates: list[str] | None = None,
                  min_confidence: float = 0.0) -> list[list[dict]]:
        """BFS for all paths between two concepts.
        Returns: [[{hop dict}, ...], ...]  — each path is a list of hops.
        """

    def shortest_path(self, source: str, target: str,
                      max_hops: int = 5) -> list[dict] | None:
        """Shortest path between two concepts (BFS)."""

    def find_common_ancestor(self, concepts: list[str],
                             predicate: str = "is_a",
                             max_hops: int = 5) -> list[dict]:
        """Find common ancestors via hierarchy predicate.
        Returns: [{"ancestor": str, "paths": [[...], ...]}, ...]
        """

    def transitive_closure(self, concept: str,
                           predicate: str,
                           max_hops: int = 5) -> set[str]:
        """All reachable concepts via a given predicate (forward)."""

    def similar_concepts(self, concept: str,
                         k: int = 10,
                         min_overlap: float = 0.3) -> list[tuple[str, float]]:
        """Concepts with similar relation profiles (Jaccard overlap).
        Returns: [(concept_name, similarity), ...]
        """

    def expand_concept(self, concept: str,
                       depth: int = 2,
                       predicates: list[str] | None = None,
                       min_confidence: float = 0.0) -> dict[str, list[dict]]:
        """Explore the concept's neighborhood up to N hops.

        Unlike query() which returns flat results, expand_concept returns
        a structured exploration: for each hop level, all concepts and
        relations found at that depth.

        Returns: {
            "concept": str,
            "levels": {
                1: [{"concept": str, "predicate": str, "confidence": float, ...}],
                2: [{"concept": str, "path": [str, str], "confidence": float, ...}],
            },
            "total_concepts": int,
            "max_depth": int,
        }
        This becomes the basis for richer answers in FACT_LOOKUP and
        MISSING_LINK — the engine can see not just direct facts but the
        surrounding knowledge graph neighborhood.
        """

All GraphOps methods return **evidence-ready** results — each entry includes
the full path/hop information needed to reconstruct the reasoning chain.

## 4. ReasoningEngine

```python
class ReasoningEngine:
    def __init__(self, graph: WorldGraph, ops: GraphOps | None = None): ...

    def reason(self, text: str) -> Answer:
        """Main entry point: parse → execute → return."""
        query = self.parser.parse(text)
        return self._execute(query)

    def _execute(self, query: StructuredQuery) -> Answer:
        """Dispatch to the appropriate intent handler."""

    # ── Intent handlers ──

    def _fact_lookup(self, query: StructuredQuery) -> Answer:
        """All facts about a concept (outgoing + incoming relations)."""

    def _relation_query(self, query: StructuredQuery) -> Answer:
        """Find targets of a relation from the subject."""
        # e.g., "What does a cat sit on?" → cat -[on]-> X

    def _reverse_query(self, query: StructuredQuery) -> Answer:
        """Find sources of a relation pointing to the object."""
        # e.g., "What sits on a table?" → X -[on]-> table

    def _hierarchy_check(self, query: StructuredQuery) -> Answer:
        """Check if subject is_a object via transitive closure."""
        # e.g., "Is a cat an animal?" → cat -[is_a*]-> animal

    def _path_finding(self, query: StructuredQuery) -> Answer:
        """Find paths between two concepts."""
        # e.g., "How is cat related to dog?"

    # ── Inference ──

    def _inherit(self, concept: str,
                 max_hops: int = 5,
                 decay: float = 0.9) -> list[InheritedFact]:
        """Inherit properties from ancestor concepts.
        
        If cat is_a mammal and mammal has fur, infer cat has fur.
        
        Contradiction resolution: if a child concept has a relation with
        the same predicate as an ancestor, the child's version wins
        (specificity overrides generality). E.g., penguin -[can]-> fly
        (polarity=False) overrides bird -[can]-> fly (polarity=True).
        
        Returns: [InheritedFact, ...] each with:
          predicate, target, confidence, evidence_chain, polarity
        """
```

### Answer format

```python
@dataclass
class Answer:
    query: str                        # original question
    intent: str                       # intent type
    results: list[Result]             # one per candidate answer
    evidence: list[list[str]]         # reasoning chains
    confidence: float                 # aggregate confidence
    inferred: bool                    # True if any result is inferred
```

```python
@dataclass
class Result:
    subject: str
    predicate: str
    target: str
    confidence: float
    count: int
    sources: list[str]
    observation_count: int            # total observations of this fact
    last_confirmed: float             # timestamp of last confirmation
    polarity: bool = True             # True = fact holds, False = negated
    inferred: bool                    # True if from inheritance, not direct
    evidence_chain: list[list[str]]   # the reasoning path
```

## 5. QueryParser

```python
class QueryParser:
    def parse(self, text: str) -> StructuredQuery: ...
```

Parsing strategy (in order):
1. **Pattern match** against question-word + dependency patterns
2. **spaCy dependency parse** for subject/predicate/object extraction
3. **Intent classification** via question-word + verb patterns:

| Question starts with | Likely intent |
|---|---|
| "what is", "what are", "tell me about" | FACT_LOOKUP |
| "what does", "what do", "what is [sub] [verb]ing" | RELATION_QUERY |
| "what [verb]s" | REVERSE_QUERY |
| "is [sub] a[n]" | HIERARCHY_CHECK |
| "how is [sub] related to" | PATH_FINDING |
| "how are [sub] and [obj] different" | COMPARISON |
| "what if" | HYPOTHESIS (scaffold) |
| "why does" | CAUSAL_QUERY (scaffold) |
| "what happens" | TEMPORAL_QUERY (scaffold) |

## 6. Formatter

```python
class Formatter:
    def format(self, answer: Answer) -> str:
        """Convert Answer to natural language string."""
```

Templates per intent:
- **FACT_LOOKUP**: "Cat has attributes: [attrs]. Cat is related to: [relations]."
- **RELATION_QUERY**: "A cat sits on: couch (95% confidence), bed (98%), table (97%)."
- **HIERARCHY_CHECK**: "Yes, a cat is an animal. (confidence: 0.92)"
- **REVERSE_QUERY**: "Things on a table: plate (99%), book (97%), cup (94%)."
- **PATH_FINDING**: "Cat is related to dog through: cat -[hunts]-> mouse -[is_food_for]-> dog"

When `inferred=True`: append "(inferred from [chain])".

## 7. Integration with Engine

```python
# In worldfield/core/engine.py

class Engine:
    def __init__(self, ...):
        ...
        self._reasoner = None
    
    @property
    def reasoner(self):
        if self._reasoner is None:
            from ..reasoning import ReasoningEngine
            self._reasoner = ReasoningEngine(self.graph)
        return self._reasoner
    
    def reason(self, text: str) -> dict:
        """Answer a natural language question."""
        return self.reasoner.reason(text).to_dict()
```

## 8. Implementation Priority

| Step | What | Why first |
|---|---|---|
| 1 | `parser.py` + `StructuredQuery` | All handlers need parsed input |
| 2 | `graph_ops.py` — `neighbors` | Foundation for all traversal |
| 3 | `graph_ops.py` — `transitive_closure`, `expand_concept` | Needed by hierarchy + inference |
| 4 | `graph_ops.py` — `find_path`, `shortest_path` | Path finding |
| 5 | `engine.py` — `FACT_LOOKUP` | Simplest handler, validates pipeline |
| 6 | `engine.py` — `RELATION_QUERY`, `REVERSE_QUERY` | Core graph queries |
| 7 | `engine.py` — `HIERARCHY_CHECK` | First non-trivial reasoning |
| 8 | `engine.py` — `_inherit()` + contradiction resolution | First real inference (not retrieval) |
| 9 | `graph_ops.py` — `find_common_ancestor`, `similar_concepts` | Comparison support |
| 10 | `engine.py` — `PATH_FINDING` + confidence propagation | Multi-hop reasoning |
| 11 | `formatter.py` | Polish — NL output |
| 12 | Integration: `Engine.reason()` | Public API |
| 13 | Inference cache hooks (reserved structures) | Future-proofing |

## 9. Confidence Propagation

When following a chain of relations, confidence propagates with decay.

**Multiplicative multiplication collapses confidence too fast** (0.95 × 0.90 ×
0.9 ≈ 0.77 after one hop; after 3 hops ≈ 0.5). Instead use:

```
P_inferred = min(edge_confidences) × decay^hops
```

Where `decay = 0.95` by default (configurable).

This preserves the weakest link in the chain while still penalizing length.

Example:
```
cat -[is_a 0.95]-> mammal -[has 0.90]-> fur
P = min(0.95, 0.90) × 0.95^1 = 0.90 × 0.95 = 0.855
```

For **multiple independent paths** supporting the same conclusion, use noisy-OR:
```
P_combined = 1 - Π(1 - P_i)
```

Multiple paths boost confidence: two paths with P=0.8 give P_combined=0.96.

For **contradictory evidence** (same predicate, different polarity):
- The most specific concept's relation wins (inheritance override)
- If equally specific, the one with higher support_count wins

## 10. Inference Cache (hooks, not implementation)

Long inheritance chains (cat → mammal → animal → living_thing) are recomputed
on every query. As the graph grows, this becomes expensive.

Reserve space for an inference cache:

```python
@dataclass
class InferenceCache:
    derived_edges: dict[str, list[dict]]  # key: "concept|predicate" → inferred targets
    last_updated: float                    # timestamp
    staleness_threshold: float = 3600     # seconds before refresh
```

The cache stores derived (inferred) edges separately from observed edges.
When a query needs them, it checks the cache first; if stale or missing,
recomputes via `_inherit()` and refreshes the cache.

**Not implemented in V1** but the architecture must allow inserting a cache
layer between GraphOps and ReasoningEngine without refactoring.

## 11. Design Decisions

1. **No external LLM.** Pure graph-based reasoning. If the graph lacks the
   answer, the engine says "I don't know" and shows what it does have.
2. **Evidence everywhere.** Every result includes the full reasoning chain.
3. **Probabilistic.** All results have confidence scores. No boolean answers.
4. **GraphOps isolation.** Reasoning engine never accesses the graph directly.
   This makes GraphOps testable and the engine swappable.
5. **Inheritance is inference.** Property inheritance via is_a chains is the
   first form of real reasoning (creating new facts from existing ones).
   Future forms: analogical mapping, causal chaining, temporal ordering.
