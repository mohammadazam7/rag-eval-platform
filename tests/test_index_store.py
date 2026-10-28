from pathlib import Path

import numpy as np
import pytest

from rag_eval.documents import Chunk
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.index_store import IndexMismatchError, load_index, save_index

TEXTS = [
    "partitioned parquet files on object storage",
    "a star schema with one fact table",
    "late arriving rows handled by a watermark",
]
CHUNKS = [Chunk(doc_id=f"d{i}", start=0, end=len(t), text=t) for i, t in enumerate(TEXTS)]


def test_round_trip_preserves_results(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=128)
    original = DenseRetriever(CHUNKS, embedder)
    save_index(original, tmp_path)
    loaded = load_index(tmp_path, embedder)
    assert loaded.chunks == original.chunks
    np.testing.assert_array_equal(loaded.matrix, original.matrix)
    query = "parquet on object storage"
    assert loaded.search(query) == original.search(query)


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    save_index(DenseRetriever(CHUNKS, HashingEmbedder(dim=128)), tmp_path)
    with pytest.raises(IndexMismatchError, match="dimension"):
        load_index(tmp_path, HashingEmbedder(dim=64))


def test_edited_chunks_are_detected(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=128)
    save_index(DenseRetriever(CHUNKS, embedder), tmp_path)
    path = tmp_path / "chunks.jsonl"
    path.write_text(path.read_text().replace("star", "snowflake"))
    with pytest.raises(IndexMismatchError, match="manifest"):
        load_index(tmp_path, embedder)
