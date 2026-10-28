import pytest

from rag_eval.documents import Chunk
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.retrieval import BM25Retriever, HybridRetriever, ScoredChunk


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, start=0, end=len(text), text=text)


CHUNKS = [
    _chunk("a", "the kafka consumer commits offsets after processing each batch"),
    _chunk("b", "spark structured streaming reads from kafka topics"),
    _chunk("c", "airflow schedules the nightly dbt run"),
    _chunk("d", "offsets are stored in a compacted internal topic"),
]


class _Fixed:
    def __init__(self, order: list[Chunk]) -> None:
        self.order = order

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return [(c, 1.0) for c in self.order[:k]]


def test_hybrid_prefers_chunks_ranked_well_by_both() -> None:
    a, b, c, d = CHUNKS
    hybrid = HybridRetriever([_Fixed([a, b]), _Fixed([d, b])], depth=2)
    ids = [chunk.doc_id for chunk, _ in hybrid.search("q", k=3)]
    assert ids[0] == "b"
    assert c not in [chunk for chunk, _ in hybrid.search("q", k=4)]


def test_hybrid_combines_bm25_and_dense() -> None:
    hybrid = HybridRetriever(
        [BM25Retriever(CHUNKS), DenseRetriever(CHUNKS, HashingEmbedder(dim=256))]
    )
    hits = hybrid.search("kafka offsets", k=2)
    assert {chunk.doc_id for chunk, _ in hits} <= {"a", "b", "d"}
    assert hits[0][0].doc_id == "a"


def test_hybrid_validates_arguments() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever(CHUNKS)], depth=0)
