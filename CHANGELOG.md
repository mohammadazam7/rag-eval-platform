# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Milestone 2: retrieval. `BM25Retriever` (Lucene IDF), `DenseRetriever` behind an
  `Embedder` protocol with a dependency-free `HashingEmbedder` baseline,
  `reciprocal_rank_fusion` and a `HybridRetriever` built on it, and on-disk dense
  indexes whose manifest is checked against the chunks and embedder on load.
- Milestone 1: corpus and chunking. `Document` and `Chunk` models with stable,
  span-derived chunk IDs; `FixedWindowChunker` and `SentenceChunker`; a loader for
  Markdown and plain-text corpora with front-matter metadata.
- Project foundation: package layout, CI (ruff, mypy in strict mode, pytest on Python
  3.11 to 3.13), roadmap, issue and pull request templates.
