"""Retrievers over a fixed set of chunks.

Every retriever returns ``(chunk, score)`` pairs, best first. Scores are only comparable
within one retriever; :func:`reciprocal_rank_fusion` combines retrievers by rank for that
reason.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from rag_eval.documents import Chunk

_WORD = re.compile(r"[a-z0-9]+")

ScoredChunk = tuple[Chunk, float]


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[ScoredChunk]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Kept deliberately simple so BM25 behaviour is easy
    to reason about in tests; stemming and stop words can be layered on later."""
    return _WORD.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 with the Lucene IDF variant, which is never negative.

    ``k1`` controls term-frequency saturation and ``b`` the strength of document-length
    normalization; the defaults are the values most commonly used in the literature.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be non-negative and b must be in [0, 1]")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, index: int) -> float:
        tf, length = self._tf[index], self._len[index]
        norm = self.k1 * (1 - self.b + self.b * length / self._avg_len) if self._avg_len else 0.0
        total = 0.0
        for term in tokenize(query):
            f = tf.get(term, 0)
            if f:
                total += self._idf[term] * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        scored = [(c, self.score(query, i)) for i, c in enumerate(self.chunks)]
        hits = [(c, s) for c, s in scored if s > 0]
        hits.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return hits[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 10, c: float = 60.0
) -> list[ScoredChunk]:
    """Fuse ranked lists with reciprocal rank fusion (Cormack et al., 2009).

    Each chunk scores ``sum(1 / (c + rank))`` over the lists it appears in, with ranks
    starting at 1. Only ranks are used, so retrievers whose raw scores live on different
    scales, such as BM25 and cosine similarity, can be combined without calibration.
    """
    if c <= 0:
        raise ValueError("c must be positive")
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            by_id[chunk.chunk_id] = chunk
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (c + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(by_id[chunk_id], score) for chunk_id, score in ordered[:k]]


class HybridRetriever:
    """Runs several retrievers and fuses their rankings with reciprocal rank fusion.

    Each retriever is asked for ``depth`` candidates so that a chunk ranked moderately by
    both a lexical and a dense retriever can outrank one that only a single retriever
    likes. ``depth`` should be at least the largest ``k`` that will be requested.
    """

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 50, c: float = 60.0) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.retrievers = list(retrievers)
        self.depth = depth
        self.c = c

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        depth = max(self.depth, k)
        rankings = [r.search(query, k=depth) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=k, c=self.c)
