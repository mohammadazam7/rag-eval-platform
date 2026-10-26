"""Document and chunk models.

A chunk keeps the character span it was cut from, so any citation can be traced back to
the exact source text, and its ID is derived from that span so it stays stable across
re-indexing as long as the document text does not change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """A source document. ``doc_id`` must be unique within a corpus."""

    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span ``[start, end)`` of a document's text."""

    doc_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid span [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError("chunk text length does not match its span")

    @property
    def chunk_id(self) -> str:
        """Stable identifier: document ID plus a short hash of the span."""
        digest = hashlib.sha1(f"{self.doc_id}:{self.start}:{self.end}".encode()).hexdigest()
        return f"{self.doc_id}#{digest[:10]}"
