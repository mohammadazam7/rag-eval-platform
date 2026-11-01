import pytest

from rag_eval.documents import Chunk
from rag_eval.generation import ExtractiveStubGenerator
from rag_eval.prompting import build_prompt, parse_answer


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, start=0, end=len(text), text=text)


AIRFLOW = _chunk("airflow.md", "The scheduler triggers the nightly dbt run at 02:00.")
KAFKA = _chunk(
    "kafka.md",
    "Consumers commit offsets after processing a batch. "
    "Committed offsets are stored in a compacted internal topic.",
)


def test_quotes_best_sentence_with_its_label() -> None:
    prompt = build_prompt("Where are committed offsets stored?", [AIRFLOW, KAFKA])
    text = ExtractiveStubGenerator(max_sentences=1).generate(prompt)
    assert text == "Committed offsets are stored in a compacted internal topic [P2]."


def test_citations_resolve_to_the_quoted_passage() -> None:
    prompt = build_prompt("When do consumers commit offsets?", [KAFKA, AIRFLOW])
    answer = parse_answer(ExtractiveStubGenerator().generate(prompt), prompt)
    assert answer.cited_chunk_ids == (KAFKA.chunk_id,)
    assert answer.unknown_labels == ()
    assert not answer.abstained


def test_abstains_without_supporting_sentence() -> None:
    prompt = build_prompt("Which region hosts the cluster?", [AIRFLOW, KAFKA])
    answer = parse_answer(ExtractiveStubGenerator().generate(prompt), prompt)
    assert answer.abstained
    assert answer.citations == ()


@pytest.mark.parametrize(
    "question,chunks", [("what is it", [KAFKA]), ("offsets", [])], ids=["stopwords", "empty"]
)
def test_abstains_on_stopword_question_or_empty_context(question: str, chunks: list[Chunk]) -> None:
    prompt = build_prompt(question, chunks)
    assert parse_answer(ExtractiveStubGenerator().generate(prompt), prompt).abstained


def test_output_is_deterministic() -> None:
    prompt = build_prompt("offsets stored topic", [KAFKA, AIRFLOW])
    gen = ExtractiveStubGenerator()
    assert gen.generate(prompt) == gen.generate(prompt)


@pytest.mark.parametrize(
    "kwargs", [{"max_sentences": 0}, {"min_overlap": 0.0}, {"min_overlap": 1.5}]
)
def test_invalid_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ExtractiveStubGenerator(**kwargs)  # type: ignore[arg-type]


def test_markdown_tables_and_markup_are_not_quoted() -> None:
    text = (
        "| ADR | Decision |\n| [0003](0003-quarantine.md) | Quarantine rejected rows |\n"
        "**Limitation.** Rejected rows are kept in quarantine for 30 days, see adr/0003.md."
    )
    prompt = build_prompt("how long are rejected rows kept", [_chunk("docs.md", text)])
    out = ExtractiveStubGenerator(max_sentences=3).generate(prompt)
    assert out == "Rejected rows are kept in quarantine for 30 days, see adr/0003.md [P1]."


def test_wrapped_lines_are_one_sentence() -> None:
    text = "There is no late-arriving-data path\nbeyond re-running a month.\n- Retries: two."
    prompt = build_prompt("late arriving data path", [_chunk("a.md", text)])
    out = ExtractiveStubGenerator(max_sentences=1).generate(prompt)
    assert out == "There is no late-arriving-data path beyond re-running a month [P1]."
