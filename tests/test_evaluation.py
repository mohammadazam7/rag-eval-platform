import json
import math
from pathlib import Path

import pytest

from rag_eval.answering import Answer, Answerer
from rag_eval.chunking import FixedWindowChunker
from rag_eval.documents import Chunk, Document
from rag_eval.evaluation.dataset import (
    GoldSetError,
    GoldSpan,
    Question,
    load_questions,
    parse_questions,
)
from rag_eval.evaluation.harness import bootstrap, compare, evaluate, ranked_chunks, report
from rag_eval.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_answer,
)
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.prompting import build_prompt, parse_answer
from rag_eval.rerank import TermProximityReranker
from rag_eval.retrieval import BM25Retriever

DOCS = {
    "tides.md": Document(
        "tides.md",
        "Tides are caused by the gravitational pull of the Moon and the Sun. "
        "Spring tides happen when the Sun and Moon align. Neap tides happen at quarter moons.",
    ),
    "bees.md": Document(
        "bees.md",
        "Honey bees communicate the direction of flowers with a waggle dance. "
        "A colony has one queen that lays the eggs.",
    ),
}


def _chunk(doc: str, start: int, end: int) -> Chunk:
    return Chunk(doc, start, end, DOCS[doc].text[start:end])


def _q(evidence: list[tuple[str, str]], qid: str = "q") -> Question:
    return parse_questions(
        [
            {
                "id": qid,
                "question": "?",
                "evidence": [{"doc_id": d, "quote": t} for d, t in evidence],
            }
        ],
        DOCS,
    )[0]


def test_quotes_resolve_to_spans_and_bad_quotes_fail(tmp_path: Path) -> None:
    q = _q([("bees.md", "waggle dance")])
    span = q.evidence[0]
    assert DOCS["bees.md"].text[span.start : span.end] == "waggle dance"
    with pytest.raises(GoldSetError, match="not found"):
        _q([("bees.md", "no such words")])
    with pytest.raises(GoldSetError, match="more than once"):
        _q([("tides.md", "tides happen")])
    with pytest.raises(GoldSetError, match="unknown document"):
        _q([("nope.md", "x")])
    path = tmp_path / "q.jsonl"
    rows = [{"id": "a", "question": "x?", "evidence": []}, {"id": "a", "question": "y?"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(GoldSetError, match="duplicate"):
        load_questions(path, DOCS)


def test_overlap_defines_relevance() -> None:
    span = GoldSpan("bees.md", 10, 20)
    assert span.overlaps(_chunk("bees.md", 15, 30))
    assert not span.overlaps(_chunk("bees.md", 20, 30))
    assert not span.overlaps(_chunk("tides.md", 10, 20))


def test_ranking_metrics() -> None:
    q = _q([("tides.md", "Spring tides happen"), ("tides.md", "Neap tides happen")])
    spring = DOCS["tides.md"].text.index("Spring")
    neap = DOCS["tides.md"].text.index("Neap")
    ranked = [
        _chunk("bees.md", 0, 20),
        _chunk("tides.md", spring, spring + 30),
        _chunk("tides.md", neap, neap + 20),
    ]
    assert recall_at_k(q, ranked, 1) == 0.0
    assert recall_at_k(q, ranked, 3) == 1.0
    assert precision_at_k(q, ranked, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(q, ranked) == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(q, ranked, 3) == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / ideal)
    unanswerable = parse_questions([{"id": "u", "question": "?"}], DOCS)[0]
    assert math.isnan(recall_at_k(unanswerable, ranked, 3))
    assert math.isnan(reciprocal_rank(unanswerable, ranked))


def test_ndcg_counts_each_gold_span_once() -> None:
    q = _q([("bees.md", "waggle dance")])
    at = DOCS["bees.md"].text.index("waggle")
    overlapping = [_chunk("bees.md", at - 10, at + 8), _chunk("bees.md", at + 2, at + 20)]
    assert ndcg_at_k(q, overlapping, 10) == 1.0
    assert ndcg_at_k(q, [_chunk("tides.md", 0, 20), *overlapping], 10) == pytest.approx(
        1 / math.log2(3)
    )


def _answer(text: str, chunks: list[Chunk]) -> Answer:
    prompt = build_prompt("q?", chunks)
    return Answer("q?", tuple((c, 1.0) for c in chunks), prompt, parse_answer(text, prompt))


def test_answer_scores() -> None:
    q = _q([("bees.md", "waggle dance")])
    good = _chunk("bees.md", 0, 69)
    other = _chunk("tides.md", 0, 40)
    s = score_answer(
        q, _answer("Honey bees communicate direction with a waggle dance [P1].", [good, other])
    )
    assert s.citation_precision == 1.0 and s.supported_share == 1.0 and not s.hallucinated
    s = score_answer(q, _answer("Bees use radar to find flowers [P2][P7].", [good, other]))
    assert s.citation_precision == 0.0 and s.fabricated_citations == 1 and s.hallucinated
    unanswerable = parse_questions([{"id": "u", "question": "?"}], DOCS)[0]
    s = score_answer(unanswerable, _answer("I don't know.", [good]))
    assert s.abstained and s.abstention_correct and not s.hallucinated
    s = score_answer(
        unanswerable, _answer("Honey bees communicate with a waggle dance [P1].", [good])
    )
    assert s.hallucinated and not s.abstention_correct


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    est = bootstrap([0.0, 1.0, 1.0, 0.0, 1.0], seed=3)
    assert est.low <= est.mean <= est.high
    assert est == bootstrap([0.0, 1.0, 1.0, 0.0, 1.0], seed=3)
    assert bootstrap([]).n == 0 and str(bootstrap([])) == "n/a"


def test_end_to_end_comparison() -> None:
    chunks = [c for d in DOCS.values() for c in FixedWindowChunker(size=8, overlap=2).chunk(d)]
    questions = parse_questions(
        [
            {
                "id": "q1",
                "question": "How do honey bees communicate the direction of flowers?",
                "evidence": [{"doc_id": "bees.md", "quote": "waggle dance"}],
            },
            {
                "id": "q2",
                "question": "When do spring tides happen?",
                "evidence": [
                    {
                        "doc_id": "tides.md",
                        "quote": "Spring tides happen when the Sun and Moon align",
                    }
                ],
            },
            {"id": "q3", "question": "What is the boiling point of mercury?"},
        ],
        DOCS,
    )
    gen = ExtractiveStubGenerator()
    plain = Answerer(BM25Retriever(chunks), gen, k=3)
    reranked = Answerer(
        BM25Retriever(chunks), gen, reranker=TermProximityReranker(), k=3, rerank_depth=10
    )
    assert 1 <= len(ranked_chunks(reranked, "honey bees", 5)) <= 5
    runs = compare({"bm25": plain, "bm25+rerank": lambda: reranked}, questions)
    assert [r.name for r in runs] == ["bm25", "bm25+rerank"]
    assert runs[0].estimate("recall@5").mean == 1.0
    table = report(runs)
    assert (
        table.startswith("| Configuration |") and "bm25+rerank" in table and "3 questions" in table
    )
    single = evaluate("x", plain, questions[:1])
    assert single.questions[0].metrics["mrr"] > 0
