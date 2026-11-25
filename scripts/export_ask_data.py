"""Export what the in-browser "Ask the manual" page needs, and a parity fixture for it.

The page answers questions without a server: BM25 retrieval and the extractive answerer
are reimplemented in TypeScript (``web/src/engine.ts``). Two files keep that honest:

- ``web/public/ask.json``: every chunk with its text and character span, the benchmark
  table parsed from ``RESULTS.md``, and a few suggested questions.
- ``web/src/parity.fixture.json``: for each of the benchmark questions, what the Python
  pipeline retrieves (top ``K`` chunk IDs and scores) and answers. A vitest test runs the
  TypeScript engine on the same questions and requires the same output.

Usage: python scripts/export_ask_data.py [CORPUS_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from rag_eval.answering import Answerer
from rag_eval.chunking import SentenceChunker
from rag_eval.documents import Chunk
from rag_eval.evaluation.dataset import load_questions
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.loaders import iter_corpus
from rag_eval.retrieval import BM25Retriever

HERE = Path(__file__).parent
ROOT = HERE.parent
BENCH = ROOT / "benchmarks" / "fastapi-tutorial"
DEFAULT_CORPUS = ROOT / "data" / "raw" / "fastapi-tutorial"
K = 4  # the answer budget the benchmark uses

SUGGESTED = ("a26", "a06", "a18", "a01")

log = logging.getLogger("export_ask_data")

_ROW = re.compile(r"^\| (?P<name>[^|]+?) \| (?P<cells>.+) \|$")
_CELL = re.compile(r"(?P<mean>[0-9.]+) \[(?P<low>[0-9.]+), (?P<high>[0-9.]+)\]")


def parse_results(text: str) -> list[dict[str, Any]]:
    """The first table of RESULTS.md as rows of ``{metric: {mean, low, high}}``."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Configuration"))
    header = [c.strip() for c in lines[start].strip("|").split("|")][1:]
    rows = []
    for line in lines[start + 2 :]:
        match = _ROW.match(line)
        if not match:
            break
        cells = [c.strip() for c in match["cells"].split("|")]
        metrics = {}
        for name, cell in zip(header, cells, strict=True):
            parsed = _CELL.fullmatch(cell)
            if parsed is None:
                raise ValueError(f"unexpected cell {cell!r} in RESULTS.md")
            metrics[name] = {k: float(parsed[k]) for k in ("mean", "low", "high")}
        rows.append({"configuration": match["name"], "metrics": metrics})
    if not rows:
        raise ValueError("no result rows found in RESULTS.md")
    return rows


def build(corpus_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    docs = {d.doc_id: d for d in iter_corpus(corpus_dir)}
    chunks: list[Chunk] = [
        c for doc in docs.values() for c in SentenceChunker(max_tokens=80).chunk(doc)
    ]
    questions = list(load_questions(BENCH / "questions.jsonl", docs))
    answerer = Answerer(BM25Retriever(chunks), ExtractiveStubGenerator(), k=K)

    fixture = []
    for q in questions:
        result = answerer.answer(q.question)
        fixture.append(
            {
                "id": q.qid,
                "question": q.question,
                "answerable": q.answerable,
                "retrieved": [
                    {"chunk_id": c.chunk_id, "score": round(s, 6)} for c, s in result.retrieved
                ],
                "answer": result.text,
                "cited": list(result.answer.cited_chunk_ids),
            }
        )

    by_id = {q.qid: q for q in questions}
    data = {
        "corpus": corpus_dir.name,
        "documents": len(docs),
        "k": K,
        "chunks": [
            {"id": c.chunk_id, "doc": c.doc_id, "start": c.start, "end": c.end, "text": c.text}
            for c in chunks
        ],
        "suggested": [by_id[i].question for i in SUGGESTED],
        "questions": {
            "total": len(questions),
            "answerable": sum(q.answerable for q in questions),
        },
        "benchmark": parse_results((BENCH / "RESULTS.md").read_text(encoding="utf-8")),
    }
    return data, fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "ask.json")
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "web" / "src" / "parity.fixture.json"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data, fixture = build(args.corpus)
    args.out.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    args.fixture.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    log.info(
        "%d chunks to %s; %d questions to %s",
        len(data["chunks"]),
        args.out,
        len(fixture),
        args.fixture,
    )


if __name__ == "__main__":
    main()
