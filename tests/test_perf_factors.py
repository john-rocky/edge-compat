"""perf_factors lane: every row validates against the registered schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((REPO / "schemas" / "perf_factor.schema.json").read_text())
ROWS = sorted((REPO / "data" / "perf_factors").glob("*.json"))


def test_lane_is_nonempty() -> None:
    assert ROWS, "data/perf_factors/ holds no rows"


@pytest.mark.parametrize("path", ROWS, ids=lambda p: p.stem)
def test_row_validates(path: Path) -> None:
    row = json.loads(path.read_text())
    jsonschema.validate(row, SCHEMA)
    assert row["id"] == path.stem, "file name must equal the row id"


@pytest.mark.parametrize("path", ROWS, ids=lambda p: p.stem)
def test_factor_consistent_with_measurement(path: Path) -> None:
    row = json.loads(path.read_text())
    m = row["measurement"]
    before, after, factor = m["before_ms"], m["after_ms"], row["factor"]
    if factor is None:
        assert before is None and after is None, (
            "null factor is reserved for feasibility-boundary rows with no measured pair"
        )
        return
    assert before is not None and after is not None
    assert factor == pytest.approx(before / after, rel=0.05), (
        f"factor {factor} disagrees with before/after {before}/{after}"
    )
