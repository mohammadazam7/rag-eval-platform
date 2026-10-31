import pytest

from rag_eval.documents import Chunk
from rag_eval.prompting import ABSTAIN, build_prompt, parse_answer


def _chunk(doc_id: str, text: str, start: int = 0) -> Chunk:
    return Chunk(doc_id=doc_id, start=start, end=start + len(text), text=text)


A = _chunk("kafka.md", "Consumers commit offsets after processing a batch.")
B = _chunk("kafka.md", "Offsets are stored in a compacted internal topic.", start=60)
C = _chunk("airflow.md", "The scheduler triggers the nightly dbt run at 02:00.")


def test_passages_are_numbered_in_rank_order() -> None:
    prompt = build_prompt("Where are offsets stored?", [B, A, C])
    assert list(prompt.passages) == ["P1", "P2", "P3"]
    assert prompt.passages["P1"] == B
    assert prompt.user.index("[P1] Offsets") < prompt.user.index("[P2] Consumers")
    assert prompt.user.endswith("Question: Where are offsets stored?")


def test_budget_skips_passages_that_do_not_fit() -> None:
    long = _chunk("long.md", " ".join(["word"] * 50))
    prompt = build_prompt("q", [A, long, C], max_context_tokens=20)
    assert [c.doc_id for c in prompt.passages.values()] == ["kafka.md", "airflow.md"]


def test_duplicates_are_included_once() -> None:
    prompt = build_prompt("q", [A, A, B])
    assert len(prompt.passages) == 2


def test_label_of_maps_chunk_ids_back() -> None:
    prompt = build_prompt("q", [A, B])
    assert prompt.label_of(B.chunk_id) == "P2"
    assert prompt.label_of(C.chunk_id) is None


def test_empty_context_is_explicit() -> None:
    assert "(no passages)" in build_prompt("q", []).user


@pytest.mark.parametrize("question,budget", [("  ", 10), ("q", 0)])
def test_invalid_arguments(question: str, budget: int) -> None:
    with pytest.raises(ValueError):
        build_prompt(question, [A], max_context_tokens=budget)


def test_parse_resolves_citations_in_first_mention_order() -> None:
    prompt = build_prompt("q", [A, B, C])
    answer = parse_answer("Offsets live in a topic [P2]. They are committed [P1][P2].", prompt)
    assert [c.label for c in answer.citations] == ["P2", "P1"]
    assert answer.cited_chunk_ids == (B.chunk_id, A.chunk_id)
    assert answer.unknown_labels == ()


def test_parse_keeps_citations_to_unseen_passages() -> None:
    prompt = build_prompt("q", [A])
    answer = parse_answer("Something [P1] and something else [P7].", prompt)
    assert [c.label for c in answer.citations] == ["P1"]
    assert answer.unknown_labels == ("P7",)


def test_parse_normalises_zero_padded_labels() -> None:
    prompt = build_prompt("q", [A])
    assert parse_answer("Yes [P01].", prompt).citations[0].label == "P1"


def test_abstention() -> None:
    prompt = build_prompt("q", [C])
    answer = parse_answer(f"  {ABSTAIN}\n", prompt)
    assert answer.abstained
    assert answer.citations == ()
