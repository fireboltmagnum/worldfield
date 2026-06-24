from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .parser import QueryParser, StructuredQuery
from .graph_ops import GraphOps
from .summarizer import ConceptSummarizer, ConceptSummary
from ..core.world_graph import WorldGraph


@dataclass
class Result:
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = 0.0
    support_count: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class Answer:
    intent: str
    question: str
    results: list[Result] = field(default_factory=list)
    summary: ConceptSummary | None = None
    query_subject: str | None = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0 and self.error is None


class ReasoningEngine:
    def __init__(self, graph: WorldGraph):
        self.graph = graph
        self.parser = QueryParser()
        self.ops = GraphOps(graph)
        self.summarizer = ConceptSummarizer(graph)

    def _build_answer(self, parsed: StructuredQuery,
                      results: list[Result] | None,
                      summary: ConceptSummary | None,
                      processing_time: float) -> Answer:
        r = results or []
        confidence = max((x.confidence for x in r), default=0.0)
        return Answer(
            intent=parsed.intent, question=parsed.raw_text,
            query_subject=parsed.subject,
            summary=summary,
            results=r, confidence=confidence,
            processing_time=processing_time,
        )

    def answer(self, question: str) -> Answer:
        t0 = time.time()
        parsed = self.parser.parse(question)
        intent = parsed.intent

        try:
            if intent == "FACT_LOOKUP":
                summary = self._handle_fact_lookup(parsed)
                return self._build_answer(parsed, [], summary, time.time() - t0)
            elif intent == "RELATION_QUERY":
                results = self._handle_relation_query(parsed)
                return self._build_answer(parsed, results, None, time.time() - t0)
            elif intent == "HIERARCHY_CHECK":
                results = self._handle_hierarchy_check(parsed)
                return self._build_answer(parsed, results, None, time.time() - t0)
            elif intent == "PATH_FINDING":
                results = self._handle_path_finding(parsed)
                return self._build_answer(parsed, results, None, time.time() - t0)
            else:
                return Answer(
                    intent=intent, question=question,
                    error=f"Unknown intent: {intent}",
                    processing_time=time.time() - t0,
                )
        except Exception as e:
            return Answer(
                intent=intent, question=question,
                error=str(e), processing_time=time.time() - t0,
            )

    # ── Intent handlers ──────────────────────────────────────────────

    def _handle_fact_lookup(self, q: StructuredQuery) -> ConceptSummary | None:
        if not q.subject:
            return None
        return self.summarizer.summarize(q.subject)

    def _handle_relation_query(self, q: StructuredQuery) -> list[Result]:
        if not q.subject:
            return []
        neighbors = self.ops.neighbors(
            q.subject, predicate=None,
            direction="outgoing", min_confidence=0.0,
        )
        results: list[Result] = []
        query_pred = (q.predicate or "").lower().strip()
        for n in neighbors:
            obj = n.get("concept", "")
            pred = n.get("predicate", "")
            conf = n.get("confidence", 0.0)
            count = n.get("count", 0)

            if query_pred and query_pred not in pred.lower():
                continue

            evidence_entry = {
                "hop": 1, "predicate": pred,
                "direction": "outgoing", "target": obj,
            }
            result = Result(
                subject=q.subject, predicate=pred, object=obj,
                confidence=conf, support_count=count,
                evidence=[evidence_entry],
                sources=n.get("sources", []),
            )
            results.append(result)
        return results

    def _handle_hierarchy_check(self, q: StructuredQuery) -> list[Result]:
        if not q.subject or not q.object:
            return []
        ancestors = self.ops.transitive_closure(q.subject, "is_a", max_hops=5)
        is_ancestor = q.object.lower() in {a.lower() for a in ancestors}

        result = Result(
            subject=q.subject,
            predicate="is_a",
            object=q.object,
            confidence=0.9 if is_ancestor else 0.0,
            support_count=1 if is_ancestor else 0,
            evidence=[{
                "check": "is_a_transitive_closure",
                "found": is_ancestor,
                "ancestors": sorted(ancestors),
            }],
        )
        return [result]

    def _handle_path_finding(self, q: StructuredQuery) -> list[Result]:
        if not q.subject or not q.object:
            return []
        paths = self.ops.find_path(q.subject, q.object, max_hops=5, min_confidence=0.0)
        if not paths:
            return []
        results: list[Result] = []
        for path in paths:
            if not path:
                continue
            avg_conf = sum(h.get("confidence", 0.0) for h in path) / len(path) if path else 0.0
            total_count = sum(h.get("count", 0) for h in path) if path else 0

            result = Result(
                subject=q.subject,
                object=q.object,
                confidence=avg_conf,
                support_count=total_count,
                evidence=path,
            )
            results.append(result)
        return results
