from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    dep: str
    tag: str
    head: int
    children: list[int] = field(default_factory=list)
    is_negated: bool = False


@dataclass
class CorefCluster:
    mentions: list[str]
    canonical: str


@dataclass
class ParsedSentence:
    text: str
    tokens: list[Token] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    noun_chunks: list[str] = field(default_factory=list)
    lemmatized: str = ""
    coref: list[CorefCluster] = field(default_factory=list)

    @property
    def nouns(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for t in self.tokens:
            if t.pos in ("NOUN", "PROPN") and t.lemma not in seen:
                seen.add(t.lemma)
                result.append(t.lemma)
        return result

    @property
    def verbs(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for t in self.tokens:
            if t.pos == "VERB" and t.lemma not in seen:
                seen.add(t.lemma)
                result.append(t.lemma)
        return result

    @property
    def adjectives(self) -> list[tuple[str, int]]:
        return [(t.lemma, t.head) for t in self.tokens if t.pos == "ADJ"]

    @property
    def amod_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for t in self.tokens:
            if t.dep == "amod" and t.head >= 0:
                noun = self.tokens[t.head]
                pairs.append((t.lemma, noun.lemma))
        return pairs

    @property
    def negated_tokens(self) -> list[int]:
        negated: set[int] = set()
        for t in self.tokens:
            if t.dep == "neg" and t.head >= 0:
                negated.add(t.head)
                # Also negate children of the negated head
                for child_idx in self.tokens[t.head].children:
                    negated.add(child_idx)
        return sorted(negated)

    @property
    def subject_verb_pairs(self) -> list[tuple[str, str, bool]]:
        pairs: list[tuple[str, str, bool]] = []
        for t in self.tokens:
            if t.dep == "nsubj" and t.head >= 0:
                verb = self.tokens[t.head]
                pairs.append((t.lemma, verb.lemma, t.is_negated))
            elif t.dep == "nsubjpass" and t.head >= 0:
                verb = self.tokens[t.head]
                pairs.append((t.lemma, verb.lemma, t.is_negated))
        return pairs

    @property
    def verb_object_pairs(self) -> list[tuple[str, str, bool]]:
        pairs: list[tuple[str, str, bool]] = []
        for t in self.tokens:
            if t.dep == "dobj" and t.head >= 0:
                verb = self.tokens[t.head]
                pairs.append((verb.lemma, t.lemma, t.is_negated))
            elif t.dep == "pobj" and t.head >= 0:
                prep_or_prt = self.tokens[t.head]
                if prep_or_prt.dep in ("prep", "prt") and prep_or_prt.head >= 0:
                    verb = self.tokens[prep_or_prt.head]
                    if verb.pos == "VERB":
                        pred = f"{verb.lemma}_{prep_or_prt.lemma}"
                        pairs.append((pred, t.lemma, t.is_negated))
        return pairs

    @property
    def triples(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        subj_map: dict[int, str] = {}
        obj_map: dict[int, str] = {}
        neg_map: dict[int, bool] = {}

        for t in self.tokens:
            if t.dep in ("nsubj", "nsubjpass") and t.head >= 0:
                subj_map[t.head] = t.lemma
                neg_map[t.head] = neg_map.get(t.head, False) or t.is_negated
            elif t.dep == "dobj" and t.head >= 0:
                obj_map[t.head] = t.lemma
                neg_map[t.head] = neg_map.get(t.head, False) or t.is_negated

        for t in self.tokens:
            if t.dep == "pobj" and t.head >= 0:
                prep_or_prt = self.tokens[t.head]
                if prep_or_prt.dep in ("prep", "prt") and prep_or_prt.head >= 0:
                    verb_idx = prep_or_prt.head
                    subj = subj_map.get(verb_idx, "")
                    verb = self.tokens[verb_idx].lemma if verb_idx < len(self.tokens) else ""
                    if subj and verb:
                        negated = neg_map.get(verb_idx, False) or t.is_negated
                        results.append({
                            "subject": subj,
                            "predicate": f"{verb}_{prep_or_prt.lemma}",
                            "object": t.lemma,
                            "negated": negated,
                        })

        for verb_idx, subj in subj_map.items():
            verb_lemma = self.tokens[verb_idx].lemma if verb_idx < len(self.tokens) else ""
            obj = obj_map.get(verb_idx)
            if obj:
                negated = neg_map.get(verb_idx, False)
                results.append({
                    "subject": subj,
                    "predicate": verb_lemma,
                    "object": obj,
                    "negated": negated,
                })

        # Passive voice: nsubjpass + agent
        for t in self.tokens:
            if t.dep == "agent" and t.head >= 0:
                verb = self.tokens[t.head]
                for child_idx in t.children:
                    child = self.tokens[child_idx]
                    if child.dep == "pobj":
                        agent = child.lemma
                        dobj = ""
                        for cd_idx in self.tokens[t.head].children:
                            cd = self.tokens[cd_idx]
                            if cd.dep == "nsubjpass":
                                dobj = cd.lemma
                        if agent and dobj:
                            negated = neg_map.get(t.head, False)
                            results.append({
                                "subject": agent,
                                "predicate": verb.lemma,
                                "object": dobj,
                                "negated": negated,
                            })

        # Copula + acomp/attr: "X is ADJ" → X is_a ADJ
        for t in self.tokens:
            if t.lemma in ("be", "is", "are", "was", "were", "am", "been", "being") and t.dep == "ROOT":
                nsubj = ""
                complement = ""
                for child_idx in t.children:
                    child = self.tokens[child_idx]
                    if child.dep == "nsubj":
                        nsubj = child.lemma
                    elif child.dep in ("acomp", "attr"):
                        complement = child.lemma
                if nsubj and complement:
                    negated = neg_map.get(t.head, False) or t.is_negated or any(
                        self.tokens[ci].is_negated for ci in t.children
                    )
                    results.append({
                        "subject": nsubj,
                        "predicate": "is_a",
                        "object": complement,
                        "negated": negated,
                    })

        return results


class NLPParser:
    """Wraps spaCy for full linguistic analysis."""

    def __init__(self):
        import spacy
        self._nlp = spacy.load("en_core_web_sm")

    def parse(self, text: str) -> ParsedSentence:
        doc = self._nlp(text)
        negated_heads = self._find_negation(doc)

        tokens: list[Token] = []
        for token in doc:
            children = [c.i for c in token.children]
            t = Token(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                dep=token.dep_,
                tag=token.tag_,
                head=token.head.i if token.head.i != token.i else -1,
                children=children,
                is_negated=token.i in negated_heads or token.dep == "neg",
            )
            tokens.append(t)

        entities = [
            {"text": ent.text, "label": ent.label_,
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

        lemmatized = " ".join(t.lemma_ for t in doc)

        coref = self._resolve_coref(doc, tokens)

        return ParsedSentence(
            text=text,
            tokens=tokens,
            entities=entities,
            noun_chunks=noun_chunks,
            lemmatized=lemmatized,
            coref=coref,
        )

    def _find_negation(self, doc) -> set[int]:
        """Find token indices affected by negation."""
        negated: set[int] = set()
        for token in doc:
            if token.dep_ == "neg":
                negated.add(token.head.i)
                for child in token.head.children:
                    negated.add(child.i)
            if token.lower_ in ("no", "never", "nothing", "nobody", "nowhere"):
                negated.add(token.i)
                if token.head:
                    negated.add(token.head.i)
        return negated

    def _resolve_coref(self, doc, tokens: list[Token]) -> list[CorefCluster]:
        """Rule-based coreference resolution (cross-sentence).

        Heuristic: pronouns in a new sentence refer to the subject (first
        NOUN/PROPN) of the immediately preceding sentence, or to the most
        recent same-sentence entity.
        """
        clusters: list[CorefCluster] = []
        pronoun_map: dict[str, str] = {}
        prev_sentence_topic: str = ""

        for sent in doc.sents:
            sent_entities: list[str] = []
            sent_topic: str = ""
            for token in sent:
                if token.pos_ in ("NOUN", "PROPN") and token.dep_ not in ("det", "attr"):
                    sent_entities.append(token.lemma_.lower())
                    if not sent_topic and token.dep_ == "nsubj":
                        sent_topic = token.lemma_.lower()
            # Fallback topic: first entity in sentence
            if not sent_topic and sent_entities:
                sent_topic = sent_entities[0]

            for token in sent:
                if token.pos_ == "PRON":
                    pronouns = set(("it", "he", "she", "they", "this", "that", "these", "those"))
                    if token.lemma_.lower() in pronouns:
                        antecedent = ""
                        if sent_entities:
                            antecedent = sent_entities[-1]
                        elif prev_sentence_topic:
                            antecedent = prev_sentence_topic
                        if antecedent:
                            pronoun_map[token.lemma_.lower()] = antecedent

            if sent_topic:
                prev_sentence_topic = sent_topic

        for pron, antecedent in pronoun_map.items():
            clusters.append(CorefCluster(
                mentions=[antecedent, pron],
                canonical=antecedent,
            ))

        return clusters
