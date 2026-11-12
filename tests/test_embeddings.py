from __future__ import annotations

import numpy as np
import pytest

from rag_eval.documents import Chunk
from rag_eval.embeddings import DenseRetriever, HashingEmbedder, WordVectorEmbedder


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(doc_id, 0, len(text), text)


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed(["spark streaming jobs", ""])
    b = embedder.embed(["spark streaming jobs", ""])
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0)
    assert not a[1].any()


def test_similar_texts_score_higher_than_unrelated_ones() -> None:
    embedder = HashingEmbedder()
    q, near, far = embedder.embed(
        ["incremental refresh policy", "configure an incremental refresh policy", "kafka"]
    )
    assert q @ near > q @ far


def test_dense_retriever_ranks_by_similarity() -> None:
    chunks = [
        _chunk("a", "Redshift COPY loads parquet from S3"),
        _chunk("b", "Athena queries the Glue catalog"),
    ]
    hits = DenseRetriever(chunks, HashingEmbedder()).search("load parquet into Redshift", k=1)
    assert hits[0][0].doc_id == "a"


def test_dense_retriever_handles_empty_corpus() -> None:
    assert DenseRetriever([], HashingEmbedder()).search("anything") == []


def test_embedder_rejects_tiny_dimension() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)


VECTORS = {
    "bee": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "hive": np.array([0.9, 0.1, 0.0], dtype=np.float32),
    "tide": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def test_word_vector_embedder_averages_known_tokens_only() -> None:
    embedder = WordVectorEmbedder(VECTORS.get, dim=3)
    out = embedder.embed(["bee hive zzz", "zzz", "tide"])
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert out[0][0] > 0.99 and out[0][2] == 0.0
    assert not out[1].any()  # nothing known: zeros, so it matches nothing
    assert np.allclose(out[2], [0.0, 0.0, 1.0])


def test_word_vectors_find_related_words_that_hashing_cannot() -> None:
    chunks = [
        Chunk(doc_id="a.md", start=0, end=4, text="hive"),
        Chunk(doc_id="b.md", start=0, end=4, text="tide"),
    ]
    retriever = DenseRetriever(chunks, WordVectorEmbedder(VECTORS.get, dim=3))
    [(best, _)] = retriever.search("bee", k=1)
    assert best.doc_id == "a.md"
