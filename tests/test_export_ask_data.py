from __future__ import annotations

import pytest
from export_ask_data import parse_results

TABLE = """# Results

| Configuration | recall@5 | mrr |
|---|---|---|
| BM25 | 0.806 [0.667, 0.917] | 0.631 [0.508, 0.746] |
| Hybrid (RRF) | 0.750 [0.611, 0.889] | 0.524 [0.400, 0.652] |

44 questions.

| Configuration | ms per question |
|---|---|
| BM25 | 6.3 |
"""


def test_parses_the_first_table_with_its_intervals() -> None:
    rows = parse_results(TABLE)
    assert [r["configuration"] for r in rows] == ["BM25", "Hybrid (RRF)"]
    assert rows[0]["metrics"]["recall@5"] == {"mean": 0.806, "low": 0.667, "high": 0.917}
    assert rows[1]["metrics"]["mrr"]["high"] == 0.652


def test_a_cell_without_an_interval_is_an_error() -> None:
    broken = TABLE.replace("0.806 [0.667, 0.917]", "0.806")
    with pytest.raises(ValueError, match="unexpected cell"):
        parse_results(broken)


def test_a_file_without_result_rows_is_an_error() -> None:
    with pytest.raises(ValueError, match="no result rows"):
        parse_results("| Configuration | recall@5 |\n|---|---|\n\ntext\n")
