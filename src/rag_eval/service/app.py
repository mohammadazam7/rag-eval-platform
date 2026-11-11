"""FastAPI app over the retrieval and answering pipeline.

Corpora are read only from below one data root (``RAG_EVAL_DATA_ROOT``, default
``data/raw``), so a request can name a folder but never point the server at an arbitrary
path. Indexes live in memory for the life of the process; that is enough for inspecting
a benchmark and keeps the service free of a database.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_eval.answering import Answerer
from rag_eval.chunking import SentenceChunker
from rag_eval.documents import Document
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.loaders import iter_corpus
from rag_eval.rerank import TermProximityReranker
from rag_eval.retrieval import BM25Retriever, HybridRetriever, Retriever

log = logging.getLogger(__name__)

RetrieverName = Literal["bm25", "dense", "hybrid"]


@dataclass
class Corpus:
    name: str
    documents: dict[str, Document]
    retrievers: dict[str, Retriever]
    chunk_count: int
    indexed_at: float = field(default_factory=time.time)


class IndexRequest(BaseModel):
    max_tokens: int = Field(80, ge=16, le=1024)


class CorpusInfo(BaseModel):
    name: str
    documents: int
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retriever: RetrieverName = "hybrid"
    rerank: bool = False
    k: int = Field(4, ge=1, le=20)


class CitationOut(BaseModel):
    label: str
    doc_id: str
    start: int
    end: int
    text: str


class PassageOut(BaseModel):
    doc_id: str
    start: int
    end: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    citations: list[CitationOut]
    unknown_labels: list[str]
    retrieved: list[PassageOut]


def _resolve(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_dir():
        raise HTTPException(404, f"no corpus folder named {name!r} under the data root")
    return path


def create_app(data_root: Path | None = None) -> FastAPI:
    root = data_root or Path(os.environ.get("RAG_EVAL_DATA_ROOT", "data/raw"))
    corpora: dict[str, Corpus] = {}
    app = FastAPI(title="rag-eval", version="0.5.0")

    def corpus(name: str) -> Corpus:
        if name not in corpora:
            raise HTTPException(404, f"corpus {name!r} is not indexed; POST /corpora/{name}")
        return corpora[name]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/corpora")
    def list_corpora() -> list[CorpusInfo]:
        return [
            CorpusInfo(name=c.name, documents=len(c.documents), chunks=c.chunk_count)
            for c in corpora.values()
        ]

    @app.post("/corpora/{name}", status_code=201)
    def index_corpus(name: str, request: IndexRequest | None = None) -> CorpusInfo:
        settings = request or IndexRequest()
        docs = list(iter_corpus(_resolve(root, name)))
        if not docs:
            raise HTTPException(422, f"corpus {name!r} has no readable documents")
        chunker = SentenceChunker(max_tokens=settings.max_tokens)
        chunks = [c for d in docs for c in chunker.chunk(d)]
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, HashingEmbedder(512))
        corpora[name] = Corpus(
            name=name,
            documents={d.doc_id: d for d in docs},
            retrievers={"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])},
            chunk_count=len(chunks),
        )
        log.info("indexed %s: %d documents, %d chunks", name, len(docs), len(chunks))
        return CorpusInfo(name=name, documents=len(docs), chunks=len(chunks))

    @app.post("/corpora/{name}/ask")
    def ask(name: str, request: AskRequest) -> AskResponse:
        c = corpus(name)
        answerer = Answerer(
            c.retrievers[request.retriever],
            ExtractiveStubGenerator(),
            reranker=TermProximityReranker() if request.rerank else None,
            k=request.k,
        )
        result = answerer.answer(request.question)
        return AskResponse(
            question=request.question,
            answer=result.text,
            abstained=result.answer.abstained,
            citations=[
                CitationOut(
                    label=cit.label,
                    doc_id=cit.chunk.doc_id,
                    start=cit.chunk.start,
                    end=cit.chunk.end,
                    text=cit.chunk.text,
                )
                for cit in result.answer.citations
            ],
            unknown_labels=list(result.answer.unknown_labels),
            retrieved=[
                PassageOut(doc_id=ch.doc_id, start=ch.start, end=ch.end, score=float(score))
                for ch, score in result.retrieved
            ],
        )

    return app
