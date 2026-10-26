"""Chunking strategies.

Both chunkers measure size in whitespace-delimited tokens, which is a close enough proxy
for model tokens to compare strategies, and return chunks whose ``text`` is an exact
slice of the source document.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol

from rag_eval.documents import Chunk, Document

_TOKEN = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _slice(document: Document, start: int, end: int) -> Chunk:
    return Chunk(document.doc_id, start, end, document.text[start:end])


class FixedWindowChunker:
    """Sliding windows of ``size`` tokens, each overlapping the previous by ``overlap``."""

    def __init__(self, size: int = 200, overlap: int = 40) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        spans = _token_spans(document.text)
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        for first in range(0, len(spans), step):
            window = spans[first : first + self.size]
            chunks.append(_slice(document, window[0][0], window[-1][1]))
            if first + self.size >= len(spans):
                break
        return chunks


class SentenceChunker:
    """Pack whole sentences into chunks of at most ``max_tokens`` tokens.

    A sentence longer than ``max_tokens`` becomes its own chunk rather than being split
    mid-sentence. ``overlap_sentences`` trailing sentences are repeated at the start of
    the next chunk so that an answer spanning a boundary is retrievable from either side.
    """

    def __init__(self, max_tokens: int = 200, overlap_sentences: int = 1) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def _sentences(text: str) -> Iterator[tuple[int, int, int]]:
        """Yield ``(start, end, token_count)`` for each non-empty sentence."""
        start = 0
        for boundary in [*_SENTENCE_END.finditer(text), None]:
            end = boundary.start() if boundary else len(text)
            segment = text[start:end]
            stripped = segment.strip()
            if stripped:
                lead = len(segment) - len(segment.lstrip())
                s, e = start + lead, start + lead + len(stripped)
                yield s, e, len(_TOKEN.findall(stripped))
            if boundary:
                start = boundary.end()

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = list(self._sentences(document.text))
        chunks: list[Chunk] = []
        i = 0
        while i < len(sentences):
            j, tokens = i, 0
            while j < len(sentences) and (j == i or tokens + sentences[j][2] <= self.max_tokens):
                tokens += sentences[j][2]
                j += 1
            chunks.append(_slice(document, sentences[i][0], sentences[j - 1][1]))
            if j >= len(sentences):
                break
            i = max(i + 1, j - self.overlap_sentences)
        return chunks
