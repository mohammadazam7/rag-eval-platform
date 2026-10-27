from __future__ import annotations

import numpy as np
import pytest

from rag_eval.documents import Chunk
from rag_eval.embeddings import DenseRetriever, HashingEmbedder


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
