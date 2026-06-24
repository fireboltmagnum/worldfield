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

The existing `RelationEdge` in `world_graph.py` gets a `sources` list:

```python
@dataclass
class RelationEdge:
    source_id: str          # concept node ID (subject)
    predicate: str          # relation type
    target_id: str          # concept node ID (object)
    confidence: float       # 0.0–1.0
    support_count: int      # how many observations
    sources: list[str]      # provenance, e.g., ["gqa/12345", "coco", "user_input"]
    examples: list[str]     # textual evidence
    first_seen: float       # timestamp
    last_seen: float        # timestamp
    modality: str           # "text", "image", etc.
```

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
|---|---|---|
| `FACT_LOOKUP` | "What is a cat?" | Implement |
| `RELATION_QUERY` | "What does a cat sit on?" | Implement |
| `REVERSE_QUERY` | "What sits on a table?" | Implement |
| `HIERARCHY_CHECK` | "Is a cat an animal?" | Implement |
| `PATH_FINDING` | "How is cat related to dog?" | Implement |
| `COMPARISON` | "How are cats and dogs different?" | Scaffold |
| `ANALOGY` | "What is to cat as bone is to dog?" | Scaffold |
| `CAUSAL_QUERY` | "What causes rain?" | Scaffold |
| `TEMPORAL_QUERY` | "What happens after sunset?" | Scaffold |
| `HYPOTHESIS` | "What if all cats were dogs?" | Scaffold |

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
```

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
        Confidence: ancestor_confidence * decay^hops.
        
        Returns: [InheritedFact, ...] each with:
          predicate, target, confidence, evidence_chain
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
    inferred: bool                     # True if from inheritance, not direct
    evidence_chain: list[list[str]]    # the reasoning path
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
| 3 | `graph_ops.py` — `transitive_closure`, `find_path` | Needed by hierarchy + inference |
| 4 | `engine.py` — `FACT_LOOKUP` | Simplest handler, validates pipeline |
| 5 | `engine.py` — `RELATION_QUERY`, `REVERSE_QUERY` | Core graph queries |
| 6 | `engine.py` — `HIERARCHY_CHECK` | First non-trivial reasoning |
| 7 | `engine.py` — `_inherit()` | First real inference (not retrieval) |
| 8 | `graph_ops.py` — `find_common_ancestor`, `similar_concepts` | Comparison support |
| 9 | `engine.py` — `PATH_FINDING` + confidence propagation | Multi-hop reasoning |
| 10 | `formatter.py` | Polish — NL output |
| 11 | Integration: `Engine.reason()` | Public API |

## 9. Confidence Propagation

When following a chain of relations, confidence propagates with decay:

```
cat -[is_a]-> mammal (0.95) → mammal -[has]-> fur (0.90)
→ inferred: cat has fur with confidence 0.95 * 0.90 * decay(1 hop)
```

Formula:
```
P(A → C) = P(A → B) × P(B → C) × decay^hops
```

Where `decay = 0.9` by default (configurable).

For multiple paths supporting the same conclusion, use noisy-OR:
```
P_combined = 1 - Π(1 - P_i)
```

## 10. Design Decisions

1. **No external LLM.** Pure graph-based reasoning. If the graph lacks the
   answer, the engine says "I don't know" and shows what it does have.
2. **Evidence everywhere.** Every result includes the full reasoning chain.
3. **Probabilistic.** All results have confidence scores. No boolean answers.
4. **GraphOps isolation.** Reasoning engine never accesses the graph directly.
   This makes GraphOps testable and the engine swappable.
5. **Inheritance is inference.** Property inheritance via is_a chains is the
   first form of real reasoning (creating new facts from existing ones).
   Future forms: analogical mapping, causal chaining, temporal ordering.
