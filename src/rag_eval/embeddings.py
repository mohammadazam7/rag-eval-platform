"""Embedders and an exact dense retriever.

The :class:`Embedder` protocol lets a sentence-transformer model or a hosted embedding
API be dropped in later. :class:`HashingEmbedder` is a dependency-free baseline: it hashes
word unigrams and bigrams into a fixed number of buckets and L2-normalizes the result. It
captures lexical overlap only, but it is deterministic and fast, so the retrieval and
evaluation code paths can be tested end to end without downloading a model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
import numpy.typing as npt

from rag_eval.documents import Chunk
from rag_eval.retrieval import ScoredChunk, tokenize

Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> Matrix: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) else -1.0

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a} {b}" for a, b in pairwise(tokens)]
            for feature in features:
                index, sign = self._bucket(feature)
                out[row, index] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out


class DenseRetriever:
    """Exact inner-product search over normalized chunk embeddings."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.matrix = (
            embedder.embed([c.text for c in self.chunks])
            if self.chunks
            else np.zeros((0, embedder.dim), dtype=np.float32)
        )

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        scores = self.matrix @ self.embedder.embed([query])[0]
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
