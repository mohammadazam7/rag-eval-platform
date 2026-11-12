"""HTTP service: index a corpus, ask questions with citations, run evaluations."""

from rag_eval.service.app import create_app

__all__ = ["create_app"]
