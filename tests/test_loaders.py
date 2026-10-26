from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.loaders import iter_corpus, load_file


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "setup.md").write_text(
        "---\ntitle: Setup guide\nowner: data-platform\n---\n# Ignored heading\nInstall it.\n"
    )
    (tmp_path / "notes.txt").write_text("Plain text notes.\r\nSecond line.\n")
    (tmp_path / "faq.md").write_text("# Frequently asked questions\n\nQ. Why?\n")
    (tmp_path / "empty.md").write_text("   \n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "draft.md").write_text("Not part of the corpus.")
    return tmp_path


def test_iter_corpus_yields_supported_files_in_path_order(corpus: Path) -> None:
    assert [d.doc_id for d in iter_corpus(corpus)] == ["faq.md", "guides/setup.md", "notes.txt"]


def test_front_matter_becomes_metadata_and_is_stripped(corpus: Path) -> None:
    doc = load_file(corpus / "guides" / "setup.md", corpus)
    assert doc.metadata["title"] == "Setup guide"
    assert doc.metadata["owner"] == "data-platform"
    assert doc.text.startswith("# Ignored heading")


def test_heading_is_used_as_title_without_front_matter(corpus: Path) -> None:
    assert load_file(corpus / "faq.md", corpus).metadata["title"] == "Frequently asked questions"


def test_line_endings_are_normalized(corpus: Path) -> None:
    assert "\r" not in load_file(corpus / "notes.txt", corpus).text


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list(iter_corpus(tmp_path / "missing"))
