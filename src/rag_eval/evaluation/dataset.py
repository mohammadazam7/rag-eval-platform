"""Gold question sets.

Relevance is defined on character spans of the source documents, not on chunk IDs. A
chunk counts as relevant when it overlaps a gold span, so one question set scores any
chunking configuration, and changing the chunker never silently invalidates the gold
labels.

A question set is a JSON Lines file, one question per line::

    {"id": "q1", "question": "...", "evidence": [{"doc_id": "a.md", "quote": "..."}]}

``quote`` is copied from the document and resolved to a span when the set is loaded;
a quote that does not occur exactly once in its document is an error, so gold labels
cannot drift away from the corpus unnoticed. Questions with no evidence are unanswerable
from the corpus, and the right behaviour is to abstain.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rag_eval.documents import Chunk, Document


class GoldSetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GoldSpan:
    doc_id: str
    start: int
    end: int

    def overlaps(self, chunk: Chunk) -> bool:
        return chunk.doc_id == self.doc_id and chunk.start < self.end and self.start < chunk.end


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    question: str
    evidence: tuple[GoldSpan, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)

    def is_relevant(self, chunk: Chunk) -> bool:
        return any(span.overlaps(chunk) for span in self.evidence)

    def evidence_found(self, chunks: Iterable[Chunk]) -> int:
        """How many gold spans at least one of ``chunks`` overlaps."""
        chunks = list(chunks)
        return sum(1 for span in self.evidence if any(span.overlaps(c) for c in chunks))


def _resolve(doc: Document, quote: str, qid: str) -> GoldSpan:
    first = doc.text.find(quote)
    if not quote or first < 0:
        raise GoldSetError(f"{qid}: quote not found in {doc.doc_id}: {quote[:60]!r}")
    if doc.text.find(quote, first + 1) >= 0:
        raise GoldSetError(f"{qid}: quote occurs more than once in {doc.doc_id}: {quote[:60]!r}")
    return GoldSpan(doc.doc_id, first, first + len(quote))


def parse_questions(
    records: Iterable[Mapping[str, object]], corpus: Mapping[str, Document]
) -> list[Question]:
    questions: list[Question] = []
    seen: set[str] = set()
    for record in records:
        qid = str(record["id"])
        if qid in seen:
            raise GoldSetError(f"duplicate question id {qid}")
        seen.add(qid)
        raw = record.get("evidence") or []
        if not isinstance(raw, list):
            raise GoldSetError(f"{qid}: evidence must be a list")
        spans = []
        for item in raw:
            doc_id = str(item["doc_id"])
            if doc_id not in corpus:
                raise GoldSetError(f"{qid}: unknown document {doc_id}")
            spans.append(_resolve(corpus[doc_id], str(item["quote"]), qid))
        questions.append(Question(qid, str(record["question"]), tuple(spans)))
    return questions


def load_questions(path: Path, corpus: Mapping[str, Document]) -> list[Question]:
    """Load and validate a JSON Lines question set against ``corpus`` (doc_id to document)."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return parse_questions((json.loads(ln) for ln in lines), corpus)
