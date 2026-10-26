"""Load a directory of plain-text and Markdown files into documents.

Document IDs are POSIX paths relative to the corpus root, so they are stable across
machines and readable in citations. Markdown front matter is parsed into metadata and
removed from the text, and a leading ``# Title`` becomes the ``title`` field when no front
matter title is present.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from rag_eval.documents import Document

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.M)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[match.end() :]


def load_file(path: Path, root: Path) -> Document:
    """Read one file as UTF-8 and return it as a :class:`Document`."""
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    metadata: dict[str, str] = {"source": path.name}
    text = raw
    if path.suffix.lower() in {".md", ".markdown"}:
        front, text = _parse_front_matter(raw)
        metadata.update(front)
        if "title" not in metadata and (heading := _HEADING.search(text)):
            metadata["title"] = heading.group(1)
    return Document(doc_id=path.relative_to(root).as_posix(), text=text, metadata=metadata)


def iter_corpus(root: Path) -> Iterator[Document]:
    """Yield every supported, non-empty file under ``root`` in sorted path order."""
    if not root.is_dir():
        raise NotADirectoryError(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        document = load_file(path, root)
        if document.text.strip():
            yield document
