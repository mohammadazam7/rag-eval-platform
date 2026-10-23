# rag-eval-platform

[![CI](https://github.com/mohammadazam7/rag-eval-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadazam7/rag-eval-platform/actions/workflows/ci.yml)

Retrieval-augmented question answering over a document collection, with answers that cite
their sources, and an evaluation harness that measures whether a change to chunking,
retrieval or generation made the system better or worse.

> **Status:** early development. The project foundation is in place; see the
> [roadmap](docs/ROADMAP.md) for milestone dates. Results will be published here only
> once they have been measured.

## Overview

Most RAG prototypes are tuned by reading a handful of answers and deciding they look
right. This project treats the pipeline as a system under test. Every stage is a
swappable component with a fixed interface, and every configuration is scored on the
same question set with the same metrics, so two configurations can be compared on
numbers rather than impressions.

## Architecture

```mermaid
flowchart LR
    D[Documents] --> C[Chunker]
    C --> I1[BM25 index]
    C --> I2[Vector index]
    Q[Question] --> R[Hybrid retriever]
    I1 --> R
    I2 --> R
    R --> RR[Re-ranker]
    RR --> G[Generator]
    G --> A[Answer with citations]
    A --> E[Evaluation harness]
    QS[Question set with gold passages] --> E
    E --> M[Metrics report]
```

| Component | Responsibility |
|---|---|
| Chunker | Split documents into passages with stable IDs and character offsets back into the source |
| Retriever | BM25 and dense retrieval, fused with reciprocal rank fusion |
| Re-ranker | Rescore the top candidates with a more expensive model |
| Generator | Produce an answer constrained to the retrieved passages, citing passage IDs; a deterministic local stub backs the tests |
| Evaluation harness | Retrieval recall@k and MRR against gold passages; answer faithfulness and hallucination rate; side-by-side comparison of configurations |

## Design decisions

- **Stable passage identity.** A passage ID is derived from the document ID and the
  character span, so a citation remains valid across re-indexing as long as the source
  text does not change.
- **Offsets, not copies.** Each chunk records its character span in the source document.
  Citations can then be traced back to exact text, and faithfulness checks compare an
  answer against the original passage rather than a normalized copy.
- **Deterministic tests.** The generator interface has a local stub that produces
  reproducible output, so the unit and integration tests never call a hosted model.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
make install
make check
```

## License

MIT
