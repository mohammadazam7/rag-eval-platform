from __future__ import annotations

import pytest

from rag_eval.chunking import FixedWindowChunker, SentenceChunker
from rag_eval.documents import Chunk, Document


def _doc(text: str) -> Document:
    return Document(doc_id="d1", text=text)


def _assert_exact_slices(doc: Document, chunks: list[Chunk]) -> None:
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert doc.text[c.start : c.end] == c.text


class TestFixedWindowChunker:
    def test_windows_overlap_by_requested_tokens(self) -> None:
        doc = _doc(" ".join(f"w{i}" for i in range(10)))
        chunks = FixedWindowChunker(size=4, overlap=2).chunk(doc)
        assert [c.text.split() for c in chunks] == [
            ["w0", "w1", "w2", "w3"],
            ["w2", "w3", "w4", "w5"],
            ["w4", "w5", "w6", "w7"],
            ["w6", "w7", "w8", "w9"],
        ]
        _assert_exact_slices(doc, chunks)

    def test_every_token_is_covered(self) -> None:
        doc = _doc(" ".join(f"w{i}" for i in range(23)))
        chunks = FixedWindowChunker(size=5, overlap=1).chunk(doc)
        covered = {tok for c in chunks for tok in c.text.split()}
        assert covered == set(doc.text.split())

    def test_short_document_is_one_chunk(self) -> None:
        doc = _doc("only three tokens")
        assert [c.text for c in FixedWindowChunker(size=10, overlap=2).chunk(doc)] == [doc.text]

    def test_empty_document_has_no_chunks(self) -> None:
        assert FixedWindowChunker().chunk(_doc("   \n ")) == []

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
    def test_rejects_invalid_parameters(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            FixedWindowChunker(size=size, overlap=overlap)


class TestSentenceChunker:
    TEXT = "Alpha one two. Beta three four five. Gamma six. Delta seven eight nine ten."

    def test_packs_whole_sentences(self) -> None:
        doc = _doc(self.TEXT)
        chunks = SentenceChunker(max_tokens=7, overlap_sentences=0).chunk(doc)
        assert [c.text for c in chunks] == [
            "Alpha one two. Beta three four five.",
            "Gamma six. Delta seven eight nine ten.",
        ]
        _assert_exact_slices(doc, chunks)

    def test_overlap_repeats_trailing_sentence(self) -> None:
        doc = _doc(self.TEXT)
        chunks = SentenceChunker(max_tokens=7, overlap_sentences=1).chunk(doc)
        assert chunks[1].text.startswith("Beta three four five.")
        _assert_exact_slices(doc, chunks)

    def test_long_sentence_is_not_split(self) -> None:
        doc = _doc("Short. " + " ".join(["long"] * 30) + ". Tail.")
        chunks = SentenceChunker(max_tokens=5, overlap_sentences=0).chunk(doc)
        assert any(len(c.text.split()) == 30 for c in chunks)
        _assert_exact_slices(doc, chunks)

    def test_always_makes_progress_with_large_overlap(self) -> None:
        doc = _doc(self.TEXT)
        chunks = SentenceChunker(max_tokens=3, overlap_sentences=5).chunk(doc)
        assert len(chunks) == 4


class TestChunk:
    def test_chunk_id_is_stable_and_span_specific(self) -> None:
        a = Chunk("d1", 0, 5, "hello")
        assert a.chunk_id == Chunk("d1", 0, 5, "hello").chunk_id
        assert a.chunk_id != Chunk("d1", 1, 6, "ello!").chunk_id
        assert a.chunk_id.startswith("d1#")

    def test_rejects_mismatched_text(self) -> None:
        with pytest.raises(ValueError):
            Chunk("d1", 0, 5, "hi")

    def test_document_requires_id(self) -> None:
        with pytest.raises(ValueError):
            Document(doc_id="", text="x")
