from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class StructuredQuery:
    intent: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    question_word: str = ""
    raw_text: str = ""


def _clean(text: str) -> str:
    text = re.sub(r"\ba\b|\ban\b|\bthe\b", "", text)
    text = text.strip().rstrip("?.,!;:")
    return text


_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Most specific first
    (re.compile(r"^what does\s+(.+?)\s+(.+?)\s+(.+)", re.I), "RELATION_QUERY"),
    (re.compile(r"^what do\s+(.+?)\s+(.+?)\s+(.+)", re.I), "RELATION_QUERY"),

    (re.compile(r"^where\s+(is|are|was|were)\s+(.+)", re.I), "RELATION_QUERY"),
    (re.compile(r"^is\s+(.+?)\s+(a|an|the)\s+(.+)", re.I), "HIERARCHY_CHECK"),
    (re.compile(r"^are\s+(.+?)\s+(.+?)", re.I), "HIERARCHY_CHECK"),
    (re.compile(r"^how\s+(is|are)\s+(.+?)\s+related\s+to\s+(.+)", re.I), "PATH_FINDING"),
    (re.compile(r"^(what|tell me about)\s+(is|are|was|were)\s+(.+)", re.I), "FACT_LOOKUP"),
    (re.compile(r"^(what|tell me about)\s+(.+)", re.I), "FACT_LOOKUP"),
]


class QueryParser:
    def __init__(self):
        self._nlp = None
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        except ImportError:
            pass

    def _extract_subject(self, raw: str, fallback: str) -> str | None:
        if self._nlp:
            doc = self._nlp(raw)
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN") and token.dep_ != "punct":
                    if token.lemma_.lower() not in ("what", "who", "which", "where", "when", "why"):
                        return token.lemma_.lower()
        cleaned = _clean(fallback)
        return cleaned or None

    def parse(self, text: str) -> StructuredQuery:
        raw = text.strip()
        best: tuple[re.Match, str] | None = None
        for pattern, intent in _INTENT_PATTERNS:
            m = pattern.match(raw)
            if m:
                best = (m, intent)
                break
        if best is None:
            return StructuredQuery(intent="FACT_LOOKUP", raw_text=raw, question_word="what")
        match, intent = best
        groups = match.groups()
        qword = match.group(0).split()[0].lower() if match.group(0) else "what"

        if intent == "FACT_LOOKUP":
            subject = self._extract_subject(raw, groups[-1] if groups else raw)
            return StructuredQuery(intent=intent, subject=subject or None, question_word=qword, raw_text=raw)

        elif intent == "HIERARCHY_CHECK":
            subject = _clean(groups[0]) if len(groups) > 0 else ""
            object_ = _clean(groups[-1]) if len(groups) > 1 else ""
            return StructuredQuery(
                intent=intent, subject=subject or None,
                predicate="is_a", object=object_ or None,
                question_word="is", raw_text=raw,
            )

        elif intent == "RELATION_QUERY":
            # Extract subject and predicate from regex groups
            subject = _clean(groups[0]) if len(groups) >= 1 else ""
            predicate = _clean(groups[1]) if len(groups) >= 2 else ""
            object_raw = _clean(groups[2]) if len(groups) >= 3 else ""

            # Use spaCy to extract subject (first content noun) and predicate (ROOT token)
            if self._nlp:
                doc = self._nlp(raw)
                qwords = {"what", "who", "which", "where", "when", "why", "how"}
                root_token: str | None = None
                content_noun: str | None = None
                for token in doc:
                    if token.dep_ == "ROOT" and not root_token:
                        root_token = token.lemma_.lower()
                    if (token.pos_ in ("NOUN", "PROPN")
                            and token.lemma_.lower() not in qwords
                            and not content_noun):
                        content_noun = token.lemma_.lower()

                if content_noun:
                    subject = content_noun
                if root_token:
                    predicate = root_token

            return StructuredQuery(
                intent=intent,
                subject=subject or None,
                predicate=predicate or None,
                object=object_raw or None,
                question_word=qword, raw_text=raw,
            )

        elif intent == "PATH_FINDING":
            subject = _clean(groups[1]) if len(groups) > 1 else ""
            object_ = _clean(groups[-1]) if len(groups) > 2 else ""
            return StructuredQuery(
                intent=intent, subject=subject or None,
                object=object_ or None,
                question_word="how", raw_text=raw,
            )

        return StructuredQuery(intent="FACT_LOOKUP", raw_text=raw, question_word="what")
