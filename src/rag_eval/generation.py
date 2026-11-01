"""Answer generation behind a backend-neutral interface.

A :class:`Generator` turns a :class:`~rag_eval.prompting.Prompt` into answer text. Hosted
or local language models implement it with a thin adapter; the rest of the pipeline only
sees the text and parses citations from it.

:class:`ExtractiveStubGenerator` is the deterministic backend used by tests and by
evaluation runs that must not depend on a model. It answers by quoting the passage
sentences that best match the question and cites the passage each sentence came from,
so its citations are correct by construction. It abstains when no sentence clears the
overlap threshold. That makes it a useful reference point: a model backend should match
its citation precision while writing better answers.
"""

from __future__ import annotations

import re
from typing import Protocol

from rag_eval.prompting import ABSTAIN, Prompt
from rag_eval.retrieval import tokenize

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")

# Function words that carry no topic, so a match on them alone is not support.
_STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from"),
        *("how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was"),
        *("what", "when", "where", "which", "who", "why", "with"),
    }
)


class Generator(Protocol):
    def generate(self, prompt: Prompt) -> str: ...


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS}


class ExtractiveStubGenerator:
    """Quote up to ``max_sentences`` best-matching sentences, each with its ``[Pn]`` label.

    A sentence qualifies when it contains at least ``min_overlap`` of the question's
    content terms (stopwords removed). Sentences are ranked by overlap, then by passage
    order, then by position in the passage, so output depends only on the prompt.
    """

    def __init__(self, max_sentences: int = 2, min_overlap: float = 0.5) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if not 0.0 < min_overlap <= 1.0:
            raise ValueError("min_overlap must be in (0, 1]")
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, prompt: Prompt) -> str:
        wanted = _content_terms(prompt.question)
        if not wanted:
            return ABSTAIN
        candidates: list[tuple[float, int, int, str, str]] = []
        for rank, (label, chunk) in enumerate(prompt.passages.items()):
            for position, match in enumerate(_SENTENCE.finditer(chunk.text)):
                sentence = " ".join(match.group(0).split())
                if not sentence:
                    continue
                overlap = len(wanted & _content_terms(sentence)) / len(wanted)
                if overlap >= self.min_overlap:
                    candidates.append((overlap, rank, position, label, sentence))
        if not candidates:
            return ABSTAIN
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        picked = candidates[: self.max_sentences]
        parts = []
        for *_, label, sentence in picked:
            if sentence[-1] not in ".!?":
                sentence += "."
            parts.append(f"{sentence[:-1]} [{label}]{sentence[-1]}")
        return " ".join(parts)
