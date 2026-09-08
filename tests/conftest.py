from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def example_matrix_path() -> Path:
    return REPO_ROOT / "data" / "examples" / "matrix_example.json"


@pytest.fixture()
def example_csv_path() -> Path:
    return REPO_ROOT / "data" / "examples" / "table_example.csv"
