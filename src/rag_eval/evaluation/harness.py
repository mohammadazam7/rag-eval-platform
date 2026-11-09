"""Run question sets through answering configurations and compare them.

Each configuration is an :class:`~rag_eval.answering.Answerer`. For every question the
harness ranks ``depth`` chunks with the configuration's retrieval pipeline (first stage,
then the re-ranker if there is one) for the retrieval metrics, and answers the question
for the answer metrics. Aggregates carry 95% bootstrap confidence intervals, because a
difference between two configurations on a few dozen questions is often noise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from rag_eval.answering import Answerer
from rag_eval.documents import Chunk
from rag_eval.evaluation.dataset import Question
from rag_eval.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_answer,
)

KS: tuple[int, ...] = (1, 3, 5, 10)


def ranked_chunks(answerer: Answerer, question: str, depth: int) -> list[Chunk]:
    """The configuration's retrieval pipeline cut at ``depth`` instead of its prompt ``k``."""
    if answerer.reranker is None:
        return [c for c, _ in answerer.retriever.search(question, k=depth)]
    pool = max(depth, answerer.rerank_depth)
    candidates = [c for c, _ in answerer.retriever.search(question, k=pool)]
    return [c for c, _ in answerer.reranker.rerank(question, candidates)[:depth]]


@dataclass(frozen=True, slots=True)
class QuestionResult:
    qid: str
    answerable: bool
    metrics: dict[str, float]
    answer: str


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass
class RunResult:
    name: str
    questions: list[QuestionResult] = field(default_factory=list)

    def values(self, metric: str) -> list[float]:
        return [
            q.metrics[metric]
            for q in self.questions
            if not math.isnan(q.metrics.get(metric, math.nan))
        ]

    def estimate(self, metric: str, samples: int = 2000, seed: int = 0) -> Estimate:
        return bootstrap(self.values(metric), samples=samples, seed=seed)


def bootstrap(values: Sequence[float], samples: int = 2000, seed: int = 0) -> Estimate:
    """Mean with a 95% percentile bootstrap interval; deterministic for a given seed."""
    n = len(values)
    if n == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    mean = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    return Estimate(mean, means[int(0.025 * samples)], means[int(0.975 * samples) - 1], n)


def evaluate(
    name: str, answerer: Answerer, questions: Sequence[Question], depth: int = max(KS)
) -> RunResult:
    result = RunResult(name)
    for q in questions:
        ranked = ranked_chunks(answerer, q.question, depth)
        metrics: dict[str, float] = {}
        for k in KS:
            metrics[f"recall@{k}"] = recall_at_k(q, ranked, k)
            metrics[f"precision@{k}"] = precision_at_k(q, ranked, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(q, ranked, k)
        metrics["mrr"] = reciprocal_rank(q, ranked)
        answer = answerer.answer(q.question)
        s = score_answer(q, answer)
        metrics["citation_precision"] = s.citation_precision
        metrics["supported_share"] = s.supported_share
        metrics["fabricated_citations"] = float(s.fabricated_citations)
        metrics["hallucination"] = float(s.hallucinated)
        metrics["abstention_correct"] = float(s.abstention_correct)
        result.questions.append(QuestionResult(q.qid, q.answerable, metrics, answer.text))
    return result


REPORT_METRICS: tuple[str, ...] = (
    "recall@5",
    "mrr",
    "ndcg@10",
    "citation_precision",
    "supported_share",
    "hallucination",
    "abstention_correct",
)


def compare(
    configs: Mapping[str, Answerer] | Mapping[str, Callable[[], Answerer]],
    questions: Sequence[Question],
) -> list[RunResult]:
    """Evaluate every configuration on the same questions, in the given order."""
    runs = []
    for name, cfg in configs.items():
        answerer = cfg if isinstance(cfg, Answerer) else cfg()
        runs.append(evaluate(name, answerer, questions))
    return runs


def report(runs: Sequence[RunResult], metrics: Sequence[str] = REPORT_METRICS) -> str:
    """A Markdown table: one row per configuration, mean and 95% CI per metric."""
    head = "| Configuration | " + " | ".join(metrics) + " |"
    rule = "|---|" + "---|" * len(metrics)
    rows = [f"| {r.name} | " + " | ".join(str(r.estimate(m)) for m in metrics) + " |" for r in runs]
    n = len(runs[0].questions) if runs else 0
    note = f"\n{n} questions. Mean and 95% bootstrap interval; lower is better for hallucination."
    return "\n".join([head, rule, *rows]) + "\n" + note + "\n"
