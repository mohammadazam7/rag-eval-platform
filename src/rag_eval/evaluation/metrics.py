"""Per-question metrics for retrieval and for cited answers.

Retrieval metrics use binary relevance: a chunk is relevant when it overlaps a gold
evidence span. Answer metrics are computed without a model judge, so they are cheap,
deterministic and reproducible:

* citation precision: share of citations whose chunk overlaps gold evidence;
* fabricated citations: labels the answer cites that were never in the prompt;
* supported sentences: share of answer sentences whose content words are mostly found in
  the passages the sentence cites (a lexical faithfulness proxy, documented as such);
* hallucination: an answer to an unanswerable question, or an answer with no supported
  sentence;
* abstention correctness: abstaining exactly when the question is unanswerable.

A model-based faithfulness judge can be added behind the same result type later; the
lexical proxy stays as the baseline it is compared against.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval.answering import Answer
from rag_eval.documents import Chunk
from rag_eval.evaluation.dataset import Question

_WORD = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\[P\d+\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
    }
)


def recall_at_k(question: Question, ranked: Sequence[Chunk], k: int) -> float:
    """Share of gold evidence spans covered by the top ``k`` chunks."""
    if not question.answerable:
        return float("nan")
    return question.evidence_found(ranked[:k]) / len(question.evidence)


def precision_at_k(question: Question, ranked: Sequence[Chunk], k: int) -> float:
    top = ranked[:k]
    if not question.answerable or not top:
        return float("nan")
    return sum(question.is_relevant(c) for c in top) / len(top)


def reciprocal_rank(question: Question, ranked: Sequence[Chunk]) -> float:
    if not question.answerable:
        return float("nan")
    for i, chunk in enumerate(ranked, 1):
        if question.is_relevant(chunk):
            return 1.0 / i
    return 0.0


def ndcg_at_k(question: Question, ranked: Sequence[Chunk], k: int) -> float:
    """Binary-relevance nDCG; the ideal ranking puts one relevant chunk per gold span first."""
    if not question.answerable:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if question.is_relevant(c)
    )
    ideal_hits = min(k, len(question.evidence))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def supported_sentences(answer: Answer, threshold: float = 0.6) -> tuple[int, int]:
    """(supported, total) answer sentences; a sentence is supported when at least
    ``threshold`` of its content words appear in the passages it cites."""
    passages = answer.prompt.passages
    total = supported = 0
    for sentence in _SENTENCE.split(answer.text.strip()):
        words = _content(_CITE.sub(" ", sentence))
        if not words:
            continue
        total += 1
        labels = [m.strip("[]") for m in _CITE.findall(sentence)]
        cited = " ".join(passages[label].text for label in labels if label in passages)
        if cited and len(words & _content(cited)) / len(words) >= threshold:
            supported += 1
    return supported, total


@dataclass(frozen=True, slots=True)
class AnswerScores:
    abstained: bool
    abstention_correct: bool
    citation_precision: float
    fabricated_citations: int
    supported_share: float
    hallucinated: bool


def score_answer(question: Question, answer: Answer) -> AnswerScores:
    cited = answer.answer
    abstained = cited.abstained
    precision = (
        sum(question.is_relevant(c.chunk) for c in cited.citations) / len(cited.citations)
        if cited.citations
        else float("nan")
    )
    supported, total = (0, 0) if abstained else supported_sentences(answer)
    share = supported / total if total else float("nan")
    hallucinated = not abstained and (not question.answerable or supported == 0)
    return AnswerScores(
        abstained=abstained,
        abstention_correct=abstained == (not question.answerable),
        citation_precision=precision,
        fabricated_citations=len(cited.unknown_labels),
        supported_share=share,
        hallucinated=hallucinated,
    )
