"""Answer questions over a folder of Markdown with cited passages.

Usage: python examples/answer.py DOCS_DIR "question" ["question" ...]

Hybrid retrieval feeds 30 candidates to the term-proximity re-ranker, the top 4 go into
the prompt, and the extractive backend answers from them. Each citation is printed with
the document and character span it resolves to.
"""

import sys
from pathlib import Path

from rag_eval.answering import Answerer
from rag_eval.chunking import SentenceChunker
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.loaders import iter_corpus
from rag_eval.rerank import TermProximityReranker
from rag_eval.retrieval import BM25Retriever, HybridRetriever

root, questions = Path(sys.argv[1]), sys.argv[2:]
chunker = SentenceChunker(max_tokens=80)
chunks = [c for d in iter_corpus(root) for c in chunker.chunk(d)]
hybrid = HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, HashingEmbedder(512))])
answerer = Answerer(
    hybrid, ExtractiveStubGenerator(), reranker=TermProximityReranker(), k=4, rerank_depth=30
)

for question in questions:
    result = answerer.answer(question)
    print(f"Q: {result.question}")
    print(f"A: {result.text}")
    for citation in result.answer.citations:
        c = citation.chunk
        print(f"   [{citation.label}] {c.doc_id} [{c.start}:{c.end}]")
    print()
