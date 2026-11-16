"""Export a 3D layout of the FastAPI tutorial corpus for the Retrieval Galaxy page.

Every chunk gets a hashing embedding (the same dependency-free embedder the benchmark
uses) and those vectors are projected to 3D with PCA, so the layout reflects real lexical
structure in the corpus rather than a made-up scatter. One demo question is run through
BM25 and the hybrid (RRF) retriever so the page can highlight what each one actually
retrieves, against the question's gold evidence.

Usage: python scripts/export_gallery_data.py [CORPUS_DIR] [--out web/public/data.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval.chunking import SentenceChunker
from rag_eval.documents import Chunk
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.evaluation.dataset import load_questions
from rag_eval.loaders import iter_corpus
from rag_eval.retrieval import BM25Retriever, HybridRetriever, ScoredChunk

HERE = Path(__file__).parent
DEFAULT_CORPUS = HERE.parent / "data" / "raw" / "fastapi-tutorial"
DEFAULT_QUESTIONS = HERE.parent / "benchmarks" / "fastapi-tutorial" / "questions.jsonl"
DEFAULT_OUT = HERE.parent / "web" / "public" / "data.json"
DEMO_QUESTION_ID = "a01"

log = logging.getLogger("export_gallery_data")


def project_to_3d(vectors: np.ndarray) -> np.ndarray:
    """PCA to 3 components via SVD on centered data. Deterministic: no random init.

    Padded with zero columns when fewer than 3 components exist, which only happens
    with a handful of chunks (a small test corpus, not the real one).
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:3].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    span = np.abs(coords).max()
    return coords / span if span > 0 else coords


def _preview(text: str, length: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def _hits(hits: list[ScoredChunk]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": chunk.chunk_id, "rank": rank, "score": round(score, 4)}
        for rank, (chunk, score) in enumerate(hits, start=1)
    ]


def build(corpus_dir: Path, questions_path: Path) -> dict[str, Any]:
    docs = {d.doc_id: d for d in iter_corpus(corpus_dir)}
    chunks: list[Chunk] = [
        c for doc in docs.values() for c in SentenceChunker(max_tokens=80).chunk(doc)
    ]
    if not chunks:
        raise ValueError(f"no chunks produced from {corpus_dir}")
    log.info("%d documents, %d chunks", len(docs), len(chunks))

    embedder = HashingEmbedder(512)
    vectors = embedder.embed([c.text for c in chunks])
    coords = project_to_3d(vectors)

    questions = {q.qid: q for q in load_questions(questions_path, docs)}
    demo = questions[DEMO_QUESTION_ID]
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder)
    hybrid = HybridRetriever([bm25, dense])
    bm25_hits = bm25.search(demo.question, k=10)
    hybrid_hits = hybrid.search(demo.question, k=10)
    gold_ids = {c.chunk_id for c in chunks if demo.is_relevant(c)}

    gallery_chunks = [
        {
            "id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "preview": _preview(chunk.text),
            "gold": chunk.chunk_id in gold_ids,
        }
        for i, chunk in enumerate(chunks)
    ]
    return {
        "corpus": corpus_dir.name,
        "question": {"id": demo.qid, "text": demo.question, "gold_chunk_ids": sorted(gold_ids)},
        "chunks": gallery_chunks,
        "retrieval": {"bm25": _hits(bm25_hits), "hybrid": _hits(hybrid_hits)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = build(args.corpus, args.questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %d chunks to %s", len(data["chunks"]), args.out)


if __name__ == "__main__":
    main()
