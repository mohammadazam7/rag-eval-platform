"""Save and load retrieval indexes.

An index directory holds three files:

* ``chunks.jsonl``: one chunk per line with its document ID, span and text.
* ``embeddings.npy``: the dense matrix, row ``i`` belonging to line ``i`` of the chunks.
* ``manifest.json``: format version, chunk count, embedding dimension and a SHA-256 of
  the chunks file, checked on load so a matrix is never paired with the wrong chunks.

BM25 statistics are not stored; rebuilding them from the chunks takes milliseconds and
avoids a second format to keep in sync.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rag_eval.documents import Chunk
from rag_eval.embeddings import DenseRetriever, Embedder

FORMAT_VERSION = 1


class IndexMismatchError(ValueError):
    """Raised when an index on disk is inconsistent or does not fit the embedder."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_index(retriever: DenseRetriever, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    chunks_path = directory / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in retriever.chunks:
            record = {
                "doc_id": chunk.doc_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.save(directory / "embeddings.npy", retriever.matrix)
    manifest = {
        "format_version": FORMAT_VERSION,
        "num_chunks": len(retriever.chunks),
        "dim": int(retriever.matrix.shape[1]),
        "chunks_sha256": _sha256(chunks_path),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_chunks(directory: Path) -> list[Chunk]:
    with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def load_index(directory: Path, embedder: Embedder) -> DenseRetriever:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise IndexMismatchError(f"unsupported index format {manifest.get('format_version')}")
    if manifest["dim"] != embedder.dim:
        raise IndexMismatchError(
            f"index has dimension {manifest['dim']} but the embedder produces {embedder.dim}"
        )
    if _sha256(directory / "chunks.jsonl") != manifest["chunks_sha256"]:
        raise IndexMismatchError("chunks.jsonl does not match the manifest")
    chunks = load_chunks(directory)
    matrix = np.load(directory / "embeddings.npy").astype(np.float32, copy=False)
    if matrix.shape != (len(chunks), embedder.dim):
        raise IndexMismatchError(f"embedding matrix has shape {matrix.shape}")
    return DenseRetriever.from_matrix(chunks, embedder, matrix)
