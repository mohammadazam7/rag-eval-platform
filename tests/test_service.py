from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_eval.service import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    corpus = tmp_path / "bees"
    corpus.mkdir()
    (corpus / "dance.md").write_text(
        "Honey bees share the location of food with a waggle dance. "
        "The angle of the dance points to the food relative to the sun.\n"
    )
    (corpus / "hive.md").write_text("A colony keeps its hive near 35 degrees Celsius.\n")
    (tmp_path / "secret.md").write_text("Outside the corpus folder.\n")
    return TestClient(create_app(tmp_path))


def test_index_then_ask_returns_cited_answer(client: TestClient) -> None:
    created = client.post("/corpora/bees")
    assert created.status_code == 201
    assert created.json() == {"name": "bees", "documents": 2, "chunks": 2}
    assert client.get("/corpora").json()[0]["name"] == "bees"

    body = client.post(
        "/corpora/bees/ask",
        json={"question": "How do bees share where food is?", "retriever": "bm25"},
    ).json()
    assert not body["abstained"]
    assert body["citations"][0]["doc_id"] == "dance.md"
    assert body["retrieved"][0]["doc_id"] == "dance.md"


def test_unrelated_question_abstains(client: TestClient) -> None:
    client.post("/corpora/bees")
    body = client.post(
        "/corpora/bees/ask", json={"question": "Which port does Redis use?", "rerank": True}
    ).json()
    assert body["abstained"]
    assert body["citations"] == []


def test_paths_outside_the_data_root_are_refused(client: TestClient) -> None:
    assert client.post("/corpora/..").status_code == 404
    assert client.post("/corpora/missing").status_code == 404


def test_asking_an_unindexed_corpus_is_a_404(client: TestClient) -> None:
    response = client.post("/corpora/bees/ask", json={"question": "anything"})
    assert response.status_code == 404
    assert "POST /corpora/bees" in response.json()["detail"]


def test_request_validation(client: TestClient) -> None:
    client.post("/corpora/bees")
    assert client.post("/corpora/bees/ask", json={"question": ""}).status_code == 422
    assert client.post("/corpora/bees/ask", json={"question": "x", "k": 0}).status_code == 422
    assert client.get("/healthz").json() == {"status": "ok"}
