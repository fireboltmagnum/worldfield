"""NLP Parser — wraps spaCy for dependency parsing and entity extraction.

Returns ParsedSentence objects that the extractor converts to concepts/relations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    dep: str
    head: int
    children: list[int] = field(default_factory=list)


@dataclass
class ParsedSentence:
    text: str
    tokens: list[Token] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    lemmatized: str = ""

    @property
    def nouns(self) -> list[str]:
        return [t.lemma for t in self.tokens if t.pos in ("NOUN", "PROPN")]

    @property
    def verbs(self) -> list[str]:
        return [t.lemma for t in self.tokens if t.pos == "VERB"]

    @property
    def subject_verb_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for t in self.tokens:
            if t.dep == "nsubj" and t.head >= 0:
                verb = self.tokens[t.head]
                pairs.append((t.lemma, verb.lemma))
        return pairs

    @property
    def verb_object_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for t in self.tokens:
            if t.dep == "dobj" and t.head >= 0:
                verb = self.tokens[t.head]
                pairs.append((verb.lemma, t.lemma))
        return pairs

    @property
    def triples(self) -> list[dict[str, str]]:
        """Extract subject-verb-(prep)-object triples."""
        results = []
        subj_map: dict[int, str] = {}
        obj_map: dict[int, str] = {}

        for t in self.tokens:
            if t.dep == "nsubj":
                subj_map[t.head] = t.lemma
            elif t.dep == "dobj":
                obj_map[t.head] = t.lemma

        for t in self.tokens:
            if t.dep == "pobj":
                # Find the preposition token that owns this pobj
                prep_token = self.tokens[t.head] if t.head < len(self.tokens) else None
                if prep_token and prep_token.dep == "prep":
                    verb_idx = prep_token.head
                    subj = subj_map.get(verb_idx, "")
                    verb = self.tokens[verb_idx].lemma if verb_idx < len(self.tokens) else ""
                    pobj = t.lemma
                    prep = prep_token.lemma
                    if subj and verb:
                        results.append({
                            "subject": subj,
                            "predicate": f"{verb}_{prep}",
                            "object": pobj,
                        })

        for verb_idx, subj in subj_map.items():
            verb_lemma = self.tokens[verb_idx].lemma if verb_idx < len(self.tokens) else ""
            obj = obj_map.get(verb_idx)
            if obj:
                results.append({
                    "subject": subj,
                    "predicate": verb_lemma,
                    "object": obj,
                })

        return results


class NLPParser:
    """Wraps spaCy. Parses text into structured representations."""

    def __init__(self):
        import spacy
        self._nlp = spacy.load("en_core_web_sm")

    def parse(self, text: str) -> ParsedSentence:
        doc = self._nlp(text)
        tokens: list[Token] = []
        for token in doc:
            children = [c.i for c in token.children]
            tokens.append(Token(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                dep=token.dep_,
                head=token.head.i if token.head.i != token.i else -1,
                children=children,
            ))
        entities = [{"text": ent.text, "label": ent.label_,
                     "start": ent.start_char, "end": ent.end_char}
                    for ent in doc.ents]

        lemmatized = " ".join(t.lemma_ for t in doc)

        return ParsedSentence(
            text=text,
            tokens=tokens,
            entities=entities,
            lemmatized=lemmatized,
        )
