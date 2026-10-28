# Roadmap

Each milestone lands on its own branch with tests and is merged when complete. Checked items are done.

- [x] **M1: Corpus and chunking** (Oct 23 to Oct 25). Document model with stable IDs,
      fixed-window and sentence-aware chunkers with overlap, character offsets back into
      the source, loaders for Markdown and plain text.
- [x] **M2: Retrieval** (Oct 26 to Oct 29). BM25 index, dense retrieval behind an
      embedding interface, reciprocal rank fusion, persisted indexes.
- [ ] **M3: Re-ranking and generation** (Oct 30 to Nov 2). Cross-encoder re-ranker
      interface, prompt construction with passage IDs, answers with citations, a
      deterministic stub backend for tests.
- [ ] **M4: Evaluation harness** (Nov 3 to Nov 6). Question sets with gold passages,
      recall@k, MRR and nDCG, citation precision, faithfulness and hallucination rate,
      comparison report across configurations.
- [ ] **M5: Service** (Nov 7 to Nov 9). FastAPI endpoints for indexing, querying and
      running evaluations, and a small web page for inspecting answers and citations.
- [ ] **M6: v1.0** (Nov 10 to Nov 12). Published results with the hardware they were
      measured on, architecture notes, CHANGELOG and release.
