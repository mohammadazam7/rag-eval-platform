"""Compare retrieval configurations on the FastAPI tutorial benchmark.

Usage: python benchmarks/fastapi-tutorial/run.py [CORPUS_DIR] [--out RESULTS.md]

Every configuration shares the chunker (sentence windows of up to 80 tokens), the
answer budget (top 4 chunks) and the extractive backend, so the rows differ only in
how candidates are retrieved and ordered.
"""

import argparse
import logging
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path

from rag_eval.answering import Answerer
from rag_eval.chunking import SentenceChunker
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.evaluation.dataset import load_questions
from rag_eval.evaluation.harness import RunResult, evaluate, report
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.loaders import iter_corpus
from rag_eval.rerank import TermProximityReranker
from rag_eval.retrieval import BM25Retriever, HybridRetriever

HERE = Path(__file__).parent
log = logging.getLogger("benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="?", type=Path, default=Path("data/raw/fastapi-tutorial"))
    parser.add_argument("--out", type=Path, default=HERE / "RESULTS.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs = list(iter_corpus(args.corpus))
    questions = load_questions(HERE / "questions.jsonl", {d.doc_id: d for d in docs})
    chunks = [c for d in docs for c in SentenceChunker(max_tokens=80).chunk(d)]
    log.info("%d documents, %d chunks, %d questions", len(docs), len(chunks), len(questions))

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, HashingEmbedder(512))
    hybrid = HybridRetriever([bm25, dense])
    gen = ExtractiveStubGenerator()
    configs: dict[str, Callable[[], Answerer]] = {
        "BM25": lambda: Answerer(bm25, gen, k=4),
        "Dense (hashing, 512)": lambda: Answerer(dense, gen, k=4),
        "Hybrid (RRF)": lambda: Answerer(hybrid, gen, k=4),
        "Hybrid + re-rank": lambda: Answerer(
            hybrid, gen, reranker=TermProximityReranker(), k=4, rerank_depth=30
        ),
    }

    runs: list[RunResult] = []
    timings: dict[str, float] = {}
    for name, make in configs.items():
        start = time.perf_counter()
        runs.append(evaluate(name, make(), questions))
        timings[name] = (time.perf_counter() - start) / len(questions) * 1000
        log.info("%s: %.1f ms per question", name, timings[name])

    answerable = sum(q.answerable for q in questions)
    lines = [
        "# FastAPI tutorial benchmark",
        "",
        f"Corpus: {len(docs)} Markdown files from the FastAPI tutorial (MIT licence), pinned in",
        f"`fetch_corpus.sh`, split into {len(chunks)} chunks. Questions: {len(questions)}",
        f"({answerable} answerable with a quoted gold span, {len(questions) - answerable}"
        " unanswerable from this corpus).",
        "",
        report(runs).rstrip(),
        "",
        "| Configuration | ms per question |",
        "|---|---|",
        *[f"| {n} | {t:.1f} |" for n, t in timings.items()],
        "",
        f"Measured on {platform.machine()}, {os.cpu_count()} vCPUs, Python"
        f" {platform.python_version()}, single process.",
        "Answers come from the extractive backend, so these rows compare retrieval and"
        " grounding, not a language model.",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
