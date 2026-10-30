"""Index a folder of Markdown and compare BM25, dense and hybrid retrieval for one query."""

import sys
from pathlib import Path

from rag_eval.chunking import SentenceChunker
from rag_eval.embeddings import DenseRetriever, HashingEmbedder
from rag_eval.loaders import iter_corpus
from rag_eval.retrieval import BM25Retriever, HybridRetriever

root, query = Path(sys.argv[1]), sys.argv[2]
docs = list(iter_corpus(root))
chunker = SentenceChunker(max_tokens=80)
chunks = [c for d in docs for c in chunker.chunk(d)]
print(f"indexed {len(docs)} documents into {len(chunks)} chunks\n")
bm25 = BM25Retriever(chunks)
dense = DenseRetriever(chunks, HashingEmbedder(dim=512))
hybrid = HybridRetriever([bm25, dense])
print(f'query: "{query}"\n')
for name, r in (("bm25", bm25), ("dense", dense), ("hybrid (rrf)", hybrid)):
    print(f"{name}")
    for rank, (c, s) in enumerate(r.search(query, k=3), 1):
        text = " ".join(c.text.split())[:78]
        print(f"  {rank}. {s:7.4f}  {c.doc_id[:34]:<34} [{c.start}:{c.end}]")
        print(f"     {text}")
    print()
