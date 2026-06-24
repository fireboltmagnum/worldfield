"""Concept Extractor — converts parsed sentences to concepts and relations.

Turns dependency parses into concept names and typed relations.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config
from .parser import ParsedSentence, NLPParser


class ConceptExtractor:
    """Extracts concepts and relations from parsed text.

    Usage:
        parser = NLPParser()
        extractor = ConceptExtractor(parser)
        concepts, relations = extractor.extract("The cat sat on the sofa")
    """

    def __init__(self, parser: NLPParser | None = None):
        self.parser = parser or NLPParser()

    def extract(self, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract concepts and relations from text.

        Returns:
            concepts: [{"name": str, "surface": str, "pos": str}, ...]
            relations: [{"source": str, "predicate": str, "target": str}, ...]
        """
        parsed = self.parser.parse(text)
        concepts = []
        relations = []
        seen_concepts = set()

        # 1. Extract named entities as concepts
        for ent in parsed.entities:
            name = ent["text"].lower()
            if name not in seen_concepts:
                concepts.append({
                    "name": name,
                    "surface": ent["text"],
                    "pos": ent["label"],
                    "is_entity": True,
                })
                seen_concepts.add(name)

        # 2. Extract nouns as concepts
        for noun in parsed.nouns:
            name = noun.lower()
            if name not in seen_concepts:
                concepts.append({
                    "name": name,
                    "surface": noun,
                    "pos": "NOUN",
                    "is_entity": False,
                })
                seen_concepts.add(name)

        # 3. Extract verbs as concepts (actions)
        for verb in parsed.verbs:
            vname = verb.lower()
            if vname not in seen_concepts:
                concepts.append({
                    "name": vname,
                    "surface": verb,
                    "pos": "VERB",
                    "is_entity": False,
                })
                seen_concepts.add(vname)

        # 4. Extract subject-verb-object triples as relations
        for triple in parsed.triples:
            subj = triple["subject"].lower()
            verb = triple["predicate"].lower()
            obj = triple["object"].lower()

            # Ensure all participants exist as concepts
            for name in [subj, obj, verb]:
                if name not in seen_concepts:
                    concepts.append({
                        "name": name,
                        "surface": name,
                        "pos": "UNKNOWN",
                        "is_entity": False,
                    })
                    seen_concepts.add(name)

            # Subject → predicate → object
            relations.append({
                "source": subj,
                "predicate": verb,
                "target": obj,
            })

            # Subject → performing → verb (agent-action)
            relations.append({
                "source": subj,
                "predicate": "performing",
                "target": verb,
            })

        # 5. Extract prepositional attachments
        for t in parsed.tokens:
            if t.dep == "prep" and t.head >= 0:
                head_lemma = parsed.tokens[t.head].lemma.lower()
                # Find the pobj child
                for child_idx in t.children:
                    child = parsed.tokens[child_idx]
                    if child.dep == "pobj":
                        obj_lemma = child.lemma.lower()
                        if head_lemma in seen_concepts and obj_lemma in seen_concepts:
                            relations.append({
                                "source": head_lemma,
                                "predicate": t.lemma.lower(),
                                "target": obj_lemma,
                            })

        return concepts, relations

    def extract_with_vectors(self, text: str, text_encoder=None,
                             cfg: Config | None = None) -> tuple[
            list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract concepts + relations, computing vectors for each concept."""
        concepts, relations = self.extract(text)
        if text_encoder is None:
            return concepts, relations

        for c in concepts:
            vec = text_encoder.encode(c["name"])
            c["vector"] = vec

        return concepts, relations
