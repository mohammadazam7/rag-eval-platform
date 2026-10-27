from __future__ import annotations

import pytest

from rag_eval.documents import Chunk
from rag_eval.retrieval import BM25Retriever, reciprocal_rank_fusion, tokenize


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(doc_id, 0, len(text), text)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        _chunk("spark", "Spark executes transformations lazily across a cluster."),
        _chunk("kafka", "Kafka stores events in partitioned, replicated logs."),
        _chunk("both", "Spark Structured Streaming can read events from Kafka topics."),
        _chunk("long", "Spark " + " ".join(["filler"] * 60)),
    ]


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("BM25, Okapi-style!") == ["bm25", "okapi", "style"]


def test_bm25_ranks_matching_chunks_first(chunks: list[Chunk]) -> None:
    hits = BM25Retriever(chunks).search("kafka events", k=2)
    assert {c.doc_id for c, _ in hits} == {"kafka", "both"}


def test_bm25_penalizes_long_chunks(chunks: list[Chunk]) -> None:
    ranked = [c.doc_id for c, _ in BM25Retriever(chunks).search("spark")]
    assert ranked.index("spark") < ranked.index("long")


def test_bm25_without_length_normalization_ties_single_mentions(chunks: list[Chunk]) -> None:
    scores = {c.doc_id: s for c, s in BM25Retriever(chunks, b=0.0).search("spark")}
    assert scores["spark"] == pytest.approx(scores["long"])


def test_bm25_returns_nothing_for_unknown_terms(chunks: list[Chunk]) -> None:
    assert BM25Retriever(chunks).search("terraform") == []


def test_bm25_handles_empty_corpus() -> None:
    assert BM25Retriever([]).search("anything") == []


def test_bm25_rejects_invalid_parameters(chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(chunks, b=1.5)


def test_rrf_rewards_agreement_between_retrievers(chunks: list[Chunk]) -> None:
    a, b, c = chunks[0], chunks[1], chunks[2]
    fused = reciprocal_rank_fusion([[(a, 9.0), (c, 5.0)], [(b, 0.9), (c, 0.8)]])
    assert fused[0][0] == c


def test_rrf_ignores_raw_score_scales(chunks: list[Chunk]) -> None:
    a, b = chunks[0], chunks[1]
    left = reciprocal_rank_fusion([[(a, 1000.0), (b, 1.0)]])
    right = reciprocal_rank_fusion([[(a, 0.2), (b, 0.1)]])
    assert [s for _, s in left] == [s for _, s in right]


def test_rrf_truncates_to_k(chunks: list[Chunk]) -> None:
    ranking = [(chunk, 1.0) for chunk in chunks]
    assert len(reciprocal_rank_fusion([ranking], k=2)) == 2
