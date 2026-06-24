from __future__ import annotations

from .engine import Answer
from .summarizer import ConceptSummary, GroupSummary, SummaryItem


def _article(word: str) -> str:
    return "an" if word and word[0] in "aeiou" else "a"


def _fmt_score(score: float) -> str:
    """Human-readable frequency descriptor."""
    if score >= 0.95:
        return "always"
    elif score >= 0.85:
        return "very often"
    elif score >= 0.70:
        return "often"
    elif score >= 0.50:
        return "sometimes"
    elif score >= 0.30:
        return "occasionally"
    else:
        return "rarely"


def _describe_item(item: SummaryItem, kind: str) -> str:
    val = item.value.replace("_", " ")
    if kind == "behavior":
        parts = val.split(" ", 1)
        if len(parts) == 2:
            verb, obj = parts
            if verb.endswith("ing"):
                verb = verb[:-3] + "s" if verb[-4] not in "aeiou" else verb[:-3] + "es"
            return f"{verb} {obj}"
        return val
    if kind == "spatial":
        return val
    return val


def _render_group(g: GroupSummary) -> str:
    if not g.items:
        return ""
    label_lower = g.label.lower()
    kind = ""
    if "behavior" in label_lower:
        kind = "behavior"
    elif "location" in label_lower:
        kind = "spatial"
    lines: list[str] = []
    for item in g.items[:5]:
        desc = _describe_item(item, kind)
        freq = _fmt_score(item.score)
        ct = item.count
        lines.append(f"  - {desc} ({freq}, {ct}x)")
    if len(g.items) > 5:
        lines.append(f"  ... and {len(g.items) - 5} more")
    return "\n".join(lines)


def _render_summary(s: ConceptSummary) -> str:
    parts = []

    for g in s.groups:
        if g.label == "Core type":
            continue
        rendered = _render_group(g)
        if not rendered:
            continue
        parts.append(f"\n{g.label}:")
        parts.append(rendered)

    return "\n".join(parts)


def format_answer(answer: Answer) -> str:
    if answer.error:
        return f"I ran into a problem: {answer.error}"

    formatters = {
        "FACT_LOOKUP": _format_fact_lookup,
        "RELATION_QUERY": _format_relation_query,
        "HIERARCHY_CHECK": _format_hierarchy_check,
        "PATH_FINDING": _format_path_finding,
    }

    fmt = formatters.get(answer.intent)
    if fmt is None:
        return f"I'm not sure how to answer that (intent={answer.intent})."
    return fmt(answer)


def _format_fact_lookup(answer: Answer) -> str:
    if answer.summary is None:
        return "I don't know anything about that yet."

    name = answer.summary.name

    # Compose a lead sentence
    parents = []
    for g in answer.summary.groups:
        if g.label == "Core type" and g.items:
            parents = [item.value for item in g.items[:3]]
    if parents:
        lead = f"{name.title()} is {_article(parents[0])} {parents[0]}."
        if len(parents) > 1:
            lead += f" Related to: {', '.join(parents[1:])}."
    else:
        lead = f"Here is what I know about {name}."

    lines = [lead, ""]

    # Attributes
    attrs = _render_summary(answer.summary)
    # The summary render already includes the name line + stats, strip that
    parts = attrs.split("\n", 1)
    if len(parts) > 1:
        lines.append(parts[1])

    # Stats
    lines.append(
        f"\n(aggregated from {answer.summary.total_observations} observations, "
        f"avg confidence {answer.summary.avg_confidence}, "
        f"{answer.processing_time:.1f}s)"
    )

    return "\n".join(lines)


def _format_relation_query(answer: Answer) -> str:
    if not answer.results:
        return "I couldn't find anything matching that."

    subj = answer.results[0].subject or "it"
    pred = answer.results[0].predicate or "..."
    lines: list[str] = [f"Things {subj} {pred}:\n"]
    for r in answer.results[:10]:
        lines.append(f"  \u2022 {r.object}  [{r.confidence:.2f}]")
    if len(answer.results) > 10:
        lines.append(f"\n  ... and {len(answer.results) - 10} more.")
    lines.append(f"\n(confidence: {answer.confidence:.2f}, {answer.processing_time:.1f}s)")
    return "\n".join(lines)


def _format_hierarchy_check(answer: Answer) -> str:
    if not answer.results:
        return "I couldn't check that hierarchy."
    r = answer.results[0]
    art = _article(r.object or "")
    if r.confidence > 0.5:
        return f"Yes, {r.subject} is {art} {r.object}."
    else:
        return f"No, {r.subject} is not {art} {r.object} (or I don't know about that relationship)."


def _format_path_finding(answer: Answer) -> str:
    if not answer.results:
        return f"I couldn't find a path between those concepts."
    lines: list[str] = [f"Paths found:\n"]
    for i, r in enumerate(answer.results[:5]):
        path_strs: list[str] = []
        for hop in r.evidence:
            src = hop.get("source", "?")
            pred = hop.get("predicate", "?")
            tgt = hop.get("target", "?")
            path_strs.append(f"{src} -[{pred}]-> {tgt}")
        full_path = "  \u2192  ".join(path_strs)
        lines.append(f"  {i + 1}. {full_path}  [{r.confidence:.2f}]")
    if len(answer.results) > 5:
        lines.append(f"\n  ... and {len(answer.results) - 5} more paths.")
    lines.append(f"\n(confidence: {answer.confidence:.2f}, {answer.processing_time:.1f}s)")
    return "\n".join(lines)
