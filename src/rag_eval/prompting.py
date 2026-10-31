"""Prompt construction with numbered passages, and parsing of the citations an answer makes.

Each retrieved chunk is shown to the model under a short label (``[P1]``, ``[P2]``, ...)
instead of its chunk ID: short labels cost fewer tokens and are copied more reliably.
:class:`Prompt` keeps the mapping from label to chunk, so a cited answer can be resolved
back to the exact document span it relies on.

Passages are added in ranked order until the context budget is spent. The budget is
counted in whitespace tokens, which is a backend-independent approximation; a backend
with its own tokenizer should leave headroom.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval.documents import Chunk

SYSTEM = (
    "Answer the question using only the numbered passages. After each sentence, cite the "
    "passages that support it in square brackets, for example [P2] or [P1][P3]. If the "
    "passages do not contain the answer, reply exactly: I don't know."
)

ABSTAIN = "I don't know."

_CITATION = re.compile(r"\[P(\d+)\]")


@dataclass(frozen=True, slots=True)
class Prompt:
    system: str
    user: str
    passages: dict[str, Chunk]

    def label_of(self, chunk_id: str) -> str | None:
        for label, chunk in self.passages.items():
            if chunk.chunk_id == chunk_id:
                return label
        return None


def _tokens(text: str) -> int:
    return len(text.split())


def build_prompt(question: str, chunks: Sequence[Chunk], max_context_tokens: int = 1500) -> Prompt:
    """Number ``chunks`` in order and pack as many as fit in ``max_context_tokens``.

    A passage that does not fit is skipped rather than truncated, since a cut passage can
    drop the sentence that made it relevant. Later, shorter passages may still fit.
    Duplicate chunks (same chunk ID) are included once.
    """
    if not question.strip():
        raise ValueError("question must be non-empty")
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")

    passages: dict[str, Chunk] = {}
    blocks: list[str] = []
    seen: set[str] = set()
    used = 0
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        cost = _tokens(chunk.text) + 1
        if used + cost > max_context_tokens:
            continue
        seen.add(chunk.chunk_id)
        label = f"P{len(passages) + 1}"
        passages[label] = chunk
        blocks.append(f"[{label}] {' '.join(chunk.text.split())}")
        used += cost

    context = "\n\n".join(blocks) if blocks else "(no passages)"
    user = f"Passages:\n\n{context}\n\nQuestion: {question.strip()}"
    return Prompt(system=SYSTEM, user=user, passages=passages)


@dataclass(frozen=True, slots=True)
class Citation:
    label: str
    chunk: Chunk


@dataclass(frozen=True, slots=True)
class CitedAnswer:
    """An answer with its citations resolved against the prompt that produced it.

    ``unknown_labels`` are labels the answer cites that were never shown to the model.
    They are kept rather than dropped because the evaluation harness counts them as
    fabricated citations.
    """

    text: str
    citations: tuple[Citation, ...]
    unknown_labels: tuple[str, ...]

    @property
    def abstained(self) -> bool:
        return self.text.strip() == ABSTAIN

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk.chunk_id for c in self.citations)


def parse_answer(text: str, prompt: Prompt) -> CitedAnswer:
    """Resolve the ``[Pn]`` markers in ``text``, in first-mention order, without duplicates."""
    citations: list[Citation] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for match in _CITATION.finditer(text):
        label = f"P{int(match.group(1))}"
        if label in seen:
            continue
        seen.add(label)
        chunk = prompt.passages.get(label)
        if chunk is None:
            unknown.append(label)
        else:
            citations.append(Citation(label, chunk))
    return CitedAnswer(text=text.strip(), citations=tuple(citations), unknown_labels=tuple(unknown))
