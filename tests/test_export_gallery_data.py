from __future__ import annotations

import numpy as np
import pytest
from export_gallery_data import _hits, _preview, build, project_to_3d

from rag_eval.documents import Chunk
from rag_eval.retrieval import BM25Retriever


def test_project_to_3d_keeps_relative_distances_and_fits_unit_cube() -> None:
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    coords = project_to_3d(vectors)

    assert coords.shape == (3, 3)
    assert np.abs(coords).max() == pytest.approx(1.0)
    # Points that were equally far apart in the input stay equally far apart.
    d01 = np.linalg.norm(coords[0] - coords[1])
    d12 = np.linalg.norm(coords[1] - coords[2])
    assert d01 == pytest.approx(d12)


def test_project_to_3d_handles_identical_vectors_without_dividing_by_zero() -> None:
    coords = project_to_3d(np.ones((4, 5)))
    assert np.allclose(coords, 0.0)


def test_preview_passes_short_text_through_and_truncates_long_text() -> None:
    assert _preview("short text") == "short text"
    long_text = "word " * 60
    preview = _preview(long_text, length=20)
    assert len(preview) <= 20
    assert preview.endswith("…")


def test_preview_collapses_internal_whitespace() -> None:
    assert _preview("a\n\n  b   c") == "a b c"


def test_hits_numbers_ranks_from_one_in_order() -> None:
    chunks = [Chunk("d", 0, 5, "hello"), Chunk("d", 6, 11, "there")]
    hits = BM25Retriever(chunks).search("hello there", k=2)
    shaped = _hits(hits)
    assert [h["rank"] for h in shaped] == [1, 2]
    assert {h["chunk_id"] for h in shaped} == {c.chunk_id for c in chunks}


def test_build_marks_the_gold_chunk_and_both_retrievers_find_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# A\n\nThe answer is forty two. It took a long time to compute.\n", encoding="utf-8"
    )
    (corpus / "b.md").write_text("# B\n\nSomething unrelated about gardening.\n", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{"id": "a01", "question": "What is the answer?", '
        '"evidence": [{"doc_id": "a.md", "quote": "The answer is forty two."}]}\n',
        encoding="utf-8",
    )

    data = build(corpus, questions)

    assert data["question"]["id"] == "a01"
    gold_ids = set(data["question"]["gold_chunk_ids"])
    assert gold_ids and all(c["gold"] for c in data["chunks"] if c["id"] in gold_ids)
    assert sum(c["gold"] for c in data["chunks"]) == len(gold_ids)
    bm25_ids = {h["chunk_id"] for h in data["retrieval"]["bm25"]}
    hybrid_ids = {h["chunk_id"] for h in data["retrieval"]["hybrid"]}
    assert gold_ids <= bm25_ids
    assert gold_ids <= hybrid_ids
