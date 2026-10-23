import rag_eval


def test_version_is_exposed() -> None:
    assert rag_eval.__version__ == "0.1.0"
