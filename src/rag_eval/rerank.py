"""Second-stage re-ranking of retrieved candidates.

First-stage retrievers score the query and each chunk independently, which keeps them
fast enough to scan a whole corpus but blind to how query terms are arranged inside a
passage. A re-ranker sees the query and one candidate together and is only run on the
top ``depth`` candidates, so it can afford a more expensive score.

:class:`Reranker` is the interface a cross-encoder model implements. The
:class:`TermProximityReranker` shipped here is a deterministic lexical baseline: it needs
no model weights, so tests and the evaluation harness can exercise the two-stage
pipeline offline, and a learned cross-encoder can later be compared against it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from rag_eval.documents import Chunk
from rag_eval.retrieval import Retriever, ScoredChunk, tokenize


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Chunk]) -> list[ScoredChunk]:
        """Score every candidate against ``query`` and return them best first."""
        ...


def _min_window(positions: dict[str, list[int]]) -> int | None:
    """Length in tokens of the shortest window containing every term at least once."""
    events = sorted((pos, term) for term, plist in positions.items() for pos in plist)
    need = len(positions)
    counts: dict[str, int] = {}
    best: int | None = None
    left = 0
    for pos, term in events:
        counts[term] = counts.get(term, 0) + 1
        while len(counts) == need:
            width = pos - events[left][0] + 1
            best = width if best is None else min(best, width)
            left_term = events[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                del counts[left_term]
            left += 1
    return best


class TermProximityReranker:
    """Score = query term coverage plus a bonus when the matched terms sit close together.

    Coverage is the fraction of distinct query terms present in the passage, weighted by
    inverse frequency across the candidate set so that rare terms count for more. The
    proximity bonus is ``alpha * matched / window``, where ``window`` is the shortest span
    of passage tokens that contains every matched term. Ties keep the incoming order,
    which makes the output deterministic for a given first-stage ranking.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha

    def rerank(self, query: str, candidates: Sequence[Chunk]) -> list[ScoredChunk]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not candidates:
            return [(c, 0.0) for c in candidates]
        tokenized = [tokenize(c.text) for c in candidates]
        n = len(candidates)
        df = {t: sum(1 for toks in tokenized if t in toks) for t in terms}
        idf = {t: math.log(1.0 + n / (1.0 + df[t])) for t in terms}
        total = sum(idf.values())

        scored: list[tuple[float, int, Chunk]] = []
        for i, (chunk, toks) in enumerate(zip(candidates, tokenized, strict=True)):
            positions: dict[str, list[int]] = {}
            for pos, tok in enumerate(toks):
                if tok in idf:
                    positions.setdefault(tok, []).append(pos)
            coverage = sum(idf[t] for t in positions) / total
            bonus = 0.0
            if len(positions) > 1:
                window = _min_window(positions)
                if window:
                    bonus = self.alpha * len(positions) / window
            scored.append((coverage + bonus, i, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(chunk, score) for score, _, chunk in scored]


class RerankingRetriever:
    """Retrieve ``depth`` candidates with ``first_stage``, re-rank them, keep the top ``k``."""

    def __init__(self, first_stage: Retriever, reranker: Reranker, depth: int = 50) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.first_stage = first_stage
        self.reranker = reranker
        self.depth = depth

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        candidates = [chunk for chunk, _ in self.first_stage.search(query, k=max(k, self.depth))]
        return self.reranker.rerank(query, candidates)[:k]
