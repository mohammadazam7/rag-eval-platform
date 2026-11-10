# FastAPI tutorial benchmark

Corpus: 51 Markdown files from the FastAPI tutorial (MIT licence), pinned in
`fetch_corpus.sh`, split into 979 chunks. Questions: 44
(36 answerable with a quoted gold span, 8 unanswerable from this corpus).

| Configuration | recall@5 | mrr | ndcg@10 | citation_precision | supported_share | hallucination | abstention_correct |
|---|---|---|---|---|---|---|---|
| BM25 | 0.806 [0.667, 0.917] | 0.631 [0.508, 0.746] | 0.706 [0.598, 0.802] | 0.520 [0.380, 0.680] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.750 [0.614, 0.886] |
| Dense (hashing, 512) | 0.528 [0.361, 0.694] | 0.424 [0.280, 0.574] | 0.479 [0.346, 0.619] | 0.500 [0.326, 0.696] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.705 [0.568, 0.841] |
| Hybrid (RRF) | 0.750 [0.611, 0.889] | 0.524 [0.400, 0.652] | 0.599 [0.484, 0.715] | 0.560 [0.400, 0.720] | 1.000 [1.000, 1.000] | 0.023 [0.000, 0.068] | 0.705 [0.568, 0.841] |
| Hybrid + re-rank | 0.667 [0.500, 0.806] | 0.559 [0.428, 0.691] | 0.630 [0.511, 0.746] | 0.500 [0.333, 0.667] | 1.000 [1.000, 1.000] | 0.023 [0.000, 0.068] | 0.614 [0.477, 0.750] |

44 questions. Mean and 95% bootstrap interval; lower is better for hallucination.

| Configuration | ms per question |
|---|---|
| BM25 | 7.9 |
| Dense (hashing, 512) | 0.9 |
| Hybrid (RRF) | 9.7 |
| Hybrid + re-rank | 10.5 |

Measured on x86_64, 2 vCPUs, Python 3.13.18, single process.
Answers come from the extractive backend, so these rows compare retrieval and grounding, not a language model.
