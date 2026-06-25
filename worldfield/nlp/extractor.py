from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config
from .parser import ParsedSentence, NLPParser

_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "myself", "yourself", "himself", "herself", "itself",
})

_COPULA_VERBS = frozenset({"be", "is", "are", "was", "were", "am", "been", "being"})


class ConceptExtractor:
    """Extracts concepts and relations from parsed text.

    Handles:
    - Compound nouns (multi-word concepts)
    - Adjective-noun attributes (has_attribute)
    - SVO triples with negation
    - Prepositional attachments
    - Passive voice
    - Copula / is-a relations
    - Coreference resolution
    """

    def __init__(self, parser: NLPParser | None = None):
        self.parser = parser or NLPParser()

    def extract(self, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parsed = self.parser.parse(text)
        concepts: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add_concept(name: str, pos: str = "NOUN",
                         surface: str = "", is_entity: bool = False):
            key = name.lower().strip()
            if key and key not in seen and key not in _PRONOUNS:
                concepts.append({
                    "name": key,
                    "surface": surface or name,
                    "pos": pos,
                    "is_entity": is_entity,
                })
                seen.add(key)

        # 1. Named entities
        for ent in parsed.entities:
            _add_concept(ent["text"].lower(), pos=ent["label"],
                         surface=ent["text"], is_entity=True)

        # 2. Multi-word noun chunks — also register the head noun
        for chunk in parsed.noun_chunks:
            cleaned = chunk.strip().lower()
            if cleaned and cleaned not in seen:
                _add_concept(cleaned, pos="NOUN", surface=chunk)
            # Register the head (last word) of each compound chunk
            head = cleaned.split()[-1] if cleaned.split() else ""
            if head and head not in seen:
                _add_concept(head, pos="NOUN", surface=head)

        # 3. Individual nouns (skip if already part of a compound chunk)
        noun_chunk_lemmas = set()
        for chunk in parsed.noun_chunks:
            for word in chunk.split():
                noun_chunk_lemmas.add(word.strip().lower())
        for noun in parsed.nouns:
            if noun not in noun_chunk_lemmas:
                _add_concept(noun, pos="NOUN", surface=noun)

        # 4. Adjectives
        for adj, _ in parsed.adjectives:
            _add_concept(adj, pos="ADJ", surface=adj)

        # 5. Adjective-noun attributes (amod)
        for adj, noun in parsed.amod_pairs:
            if adj in seen and noun in seen:
                relations.append({
                    "source": noun,
                    "predicate": "has_attribute",
                    "target": adj,
                    "negated": False,
                })

        # 6. Verbs
        for verb in parsed.verbs:
            if verb not in _COPULA_VERBS:
                _add_concept(verb, pos="VERB", surface=verb)

        # 7. SVO triples (includes copula is_a from parser)
        for triple in parsed.triples:
            subj = triple["subject"].lower()
            verb = triple["predicate"].lower()
            obj = triple["object"].lower()
            negated = triple.get("negated", False)

            for name in (subj, obj):
                if name not in _PRONOUNS:
                    _add_concept(name)
            if verb and verb not in _COPULA_VERBS and verb != "is_a":
                _add_concept(verb, pos="VERB")

            relations.append({
                "source": subj,
                "predicate": verb,
                "target": obj,
                "negated": negated,
            })

            if not negated and verb not in _COPULA_VERBS and verb != "is_a":
                relations.append({
                    "source": subj,
                    "predicate": "performing",
                    "target": verb,
                    "negated": False,
                })

        # 9. Prepositional attachments
        for t in parsed.tokens:
            if t.dep == "prep" and t.head >= 0:
                head_lemma = parsed.tokens[t.head].lemma.lower()
                for child_idx in t.children:
                    child = parsed.tokens[child_idx]
                    if child.dep == "pobj":
                        obj_lemma = child.lemma.lower()
                        if obj_lemma in _PRONOUNS or head_lemma in _PRONOUNS:
                            continue
                        _add_concept(obj_lemma)
                        already = any(
                            r["source"] == head_lemma
                            and r["predicate"] == t.lemma.lower()
                            and r["target"] == obj_lemma
                            for r in relations
                        )
                        if not already:
                            relations.append({
                                "source": head_lemma,
                                "predicate": t.lemma.lower(),
                                "target": obj_lemma,
                                "negated": child.is_negated,
                            })

        # 10. Apply coreference resolution
        for cluster in parsed.coref:
            for i, mention in enumerate(cluster.mentions):
                if mention not in seen and cluster.canonical in seen:
                    _add_concept(mention, surface=mention)
                    for r in relations:
                        if r["source"] == mention:
                            r["source"] = cluster.canonical
                        if r["target"] == mention:
                            r["target"] = cluster.canonical

        return concepts, relations

    def extract_with_vectors(self, text: str, text_encoder=None,
                             cfg: Config | None = None) -> tuple[
            list[dict[str, Any]], list[dict[str, Any]]]:
        concepts, relations = self.extract(text)
        if text_encoder is None:
            return concepts, relations
        for c in concepts:
            vec = text_encoder.encode(c["name"])
            c["vector"] = vec
        return concepts, relations
