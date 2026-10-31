import pytest

from rag_eval.documents import Chunk
from rag_eval.rerank import RerankingRetriever, TermProximityReranker, _min_window
from rag_eval.retrieval import BM25Retriever, ScoredChunk


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, start=0, end=len(text), text=text)


def test_min_window_finds_shortest_span() -> None:
    assert _min_window({"a": [0, 9], "b": [5, 10]}) == 2
    assert _min_window({"a": [3]}) == 1


def test_adjacent_terms_outrank_scattered_terms() -> None:
    near = _chunk("near", "the consumer commits offsets after each batch is processed")
    far = _chunk("far", "offsets live in a topic that the broker compacts and every consumer reads")
    ranked = TermProximityReranker().rerank("consumer offsets", [far, near])
    assert [c.doc_id for c, _ in ranked] == ["near", "far"]


def test_rare_terms_weigh_more_than_common_ones() -> None:
    common = _chunk("common", "kafka kafka kafka")
    rare = _chunk("rare", "compaction")
    other = _chunk("other", "kafka topic")
    ranked = TermProximityReranker(alpha=0.0).rerank("kafka compaction", [common, other, rare])
    assert ranked[0][0].doc_id == "rare"


def test_ties_keep_first_stage_order() -> None:
    a, b = _chunk("a", "nothing relevant"), _chunk("b", "still nothing")
    ranked = TermProximityReranker().rerank("kafka", [a, b])
    assert [c.doc_id for c, _ in ranked] == ["a", "b"]
    assert all(score == 0.0 for _, score in ranked)


def test_empty_query_or_candidates() -> None:
    reranker = TermProximityReranker()
    assert reranker.rerank("kafka", []) == []
    chunk = _chunk("a", "kafka")
    assert reranker.rerank("   ", [chunk]) == [(chunk, 0.0)]


def test_negative_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        TermProximityReranker(alpha=-0.1)


class _Recording:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.requested: list[int] = []

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        self.requested.append(k)
        return [(c, 1.0) for c in self.chunks[:k]]


def test_reranking_retriever_fetches_depth_and_truncates_to_k() -> None:
    chunks = [_chunk(str(i), f"passage {i} about kafka offsets") for i in range(8)]
    first = _Recording(chunks)
    retriever = RerankingRetriever(first, TermProximityReranker(), depth=6)
    results = retriever.search("kafka offsets", k=3)
    assert first.requested == [6]
    assert len(results) == 3


def test_reranking_retriever_over_bm25() -> None:
    chunks = [
        _chunk("a", "offsets are committed by the consumer after processing"),
        _chunk("b", "the consumer group rebalances when a member joins; offsets are stored"),
        _chunk("c", "airflow schedules the nightly dbt run"),
    ]
    retriever = RerankingRetriever(BM25Retriever(chunks), TermProximityReranker(), depth=3)
    top = retriever.search("consumer offsets committed", k=2)
    assert top[0][0].doc_id == "a"
    assert "c" not in {c.doc_id for c, _ in top}


def test_depth_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RerankingRetriever(_Recording([]), TermProximityReranker(), depth=0)
