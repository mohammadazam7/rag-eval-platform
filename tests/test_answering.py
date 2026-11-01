import pytest

from rag_eval.answering import Answerer
from rag_eval.chunking import SentenceChunker
from rag_eval.documents import Chunk, Document
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.prompting import Prompt
from rag_eval.rerank import TermProximityReranker
from rag_eval.retrieval import BM25Retriever, ScoredChunk

DOCS = [
    Document(
        "kafka.md",
        "Consumers commit offsets after processing each batch. "
        "Committed offsets are stored in a compacted internal topic. "
        "A rebalance reassigns partitions when a consumer joins or leaves the group.",
    ),
    Document(
        "airflow.md",
        "The scheduler triggers the nightly dbt run at 02:00. "
        "Failed tasks are retried twice with a five minute delay.",
    ),
]


@pytest.fixture
def chunks() -> list[Chunk]:
    chunker = SentenceChunker(max_tokens=12)
    return [c for d in DOCS for c in chunker.chunk(d)]


def test_answer_cites_the_chunk_that_contains_it(chunks: list[Chunk]) -> None:
    answerer = Answerer(BM25Retriever(chunks), ExtractiveStubGenerator(max_sentences=1), k=3)
    result = answerer.answer("How many times are failed tasks retried?")
    assert not result.answer.abstained
    (citation,) = result.answer.citations
    assert citation.chunk.doc_id == "airflow.md"
    assert "retried twice" in DOCS[1].text[citation.chunk.start : citation.chunk.end]


def test_reranker_sees_the_deeper_candidate_list(chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(chunks)
    requested: list[int] = []

    class Recording:
        def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
            requested.append(k)
            return bm25.search(query, k=k)

    answerer = Answerer(
        Recording(),
        ExtractiveStubGenerator(),
        reranker=TermProximityReranker(),
        k=2,
        rerank_depth=4,
    )
    result = answerer.answer("where are committed offsets stored")
    assert requested == [4]
    assert len(result.retrieved) == 2
    assert result.retrieved[0][0].doc_id == "kafka.md"


def test_unanswerable_question_abstains(chunks: list[Chunk]) -> None:
    answerer = Answerer(BM25Retriever(chunks), ExtractiveStubGenerator(), k=3)
    assert answerer.answer("Which cloud region hosts the warehouse?").answer.abstained


def test_prompt_reaches_the_generator(chunks: list[Chunk]) -> None:
    prompts: list[Prompt] = []

    class Echo:
        def generate(self, prompt: Prompt) -> str:
            prompts.append(prompt)
            return "See [P1] and [P9]."

    result = Answerer(BM25Retriever(chunks), Echo(), k=2).answer("  rebalance partitions  ")
    assert prompts[0].question == "rebalance partitions"
    assert result.question == "rebalance partitions"
    assert result.answer.unknown_labels == ("P9",)


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": 5, "rerank_depth": 3}])
def test_invalid_configuration(chunks: list[Chunk], kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Answerer(
            BM25Retriever(chunks),
            ExtractiveStubGenerator(),
            reranker=TermProximityReranker(),
            **kwargs,
        )
