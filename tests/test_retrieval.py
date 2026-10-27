from __future__ import annotations

import pytest

from rag_eval.documents import Chunk
from rag_eval.retrieval import BM25Retriever, tokenize


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
