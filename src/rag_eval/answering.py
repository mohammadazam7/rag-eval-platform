"""End-to-end question answering: retrieve, optionally re-rank, prompt, generate, cite.

:class:`Answerer` wires the stages together and returns everything the evaluation
harness needs to score a single question: the ranked candidates, the prompt the backend
saw, and the parsed answer with its citations resolved to chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.generation import Generator
from rag_eval.prompting import CitedAnswer, Prompt, build_prompt, parse_answer
from rag_eval.rerank import Reranker
from rag_eval.retrieval import Retriever, ScoredChunk


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    retrieved: tuple[ScoredChunk, ...]
    prompt: Prompt
    answer: CitedAnswer

    @property
    def text(self) -> str:
        return self.answer.text


class Answerer:
    """Answer questions over one corpus with a fixed configuration.

    ``k`` candidates are retrieved (``rerank_depth`` when a re-ranker is set, re-ranked
    and cut back to ``k``), then packed into a prompt of at most ``max_context_tokens``.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        reranker: Reranker | None = None,
        k: int = 5,
        rerank_depth: int = 30,
        max_context_tokens: int = 1500,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if reranker is not None and rerank_depth < k:
            raise ValueError("rerank_depth must be at least k")
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.k = k
        self.rerank_depth = rerank_depth
        self.max_context_tokens = max_context_tokens

    def retrieve(self, question: str) -> list[ScoredChunk]:
        if self.reranker is None:
            return self.retriever.search(question, k=self.k)
        candidates = [c for c, _ in self.retriever.search(question, k=self.rerank_depth)]
        return self.reranker.rerank(question, candidates)[: self.k]

    def answer(self, question: str) -> Answer:
        retrieved = self.retrieve(question)
        prompt = build_prompt(
            question, [c for c, _ in retrieved], max_context_tokens=self.max_context_tokens
        )
        text = self.generator.generate(prompt)
        return Answer(
            question=prompt.question,
            retrieved=tuple(retrieved),
            prompt=prompt,
            answer=parse_answer(text, prompt),
        )
