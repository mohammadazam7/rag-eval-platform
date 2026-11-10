# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- nDCG could exceed 1 when one gold span overlapped several chunks.

### Added
- Milestone 4: evaluation. Gold question sets whose evidence is a verbatim quote
  resolved to a character span; recall, precision and nDCG at k, MRR, citation precision,
  fabricated citations, supported-sentence share, hallucination and abstention; a harness
  that runs configurations on the same questions and reports means with bootstrap
  intervals; and a FastAPI tutorial benchmark (44 questions) with published results.
- Milestone 3: re-ranking and generation. A `Reranker` protocol with a deterministic
  `TermProximityReranker` baseline and `RerankingRetriever`; `build_prompt`, which labels
  passages `[P1]`, `[P2]`, ... within a token budget; `parse_answer`, which resolves
  citations to chunks and records labels the model was never shown; a `Generator`
  protocol with an `ExtractiveStubGenerator` that quotes supporting sentences or
  abstains; and `Answerer`, which runs the full pipeline. `examples/answer.py` shows it
  on a Markdown corpus.
- Milestone 2: retrieval. `BM25Retriever` (Lucene IDF), `DenseRetriever` behind an
  `Embedder` protocol with a dependency-free `HashingEmbedder` baseline,
  `reciprocal_rank_fusion` and a `HybridRetriever` built on it, and on-disk dense
  indexes whose manifest is checked against the chunks and embedder on load.
- Milestone 1: corpus and chunking. `Document` and `Chunk` models with stable,
  span-derived chunk IDs; `FixedWindowChunker` and `SentenceChunker`; a loader for
  Markdown and plain-text corpora with front-matter metadata.
- Project foundation: package layout, CI (ruff, mypy in strict mode, pytest on Python
  3.11 to 3.13), roadmap, issue and pull request templates.
