"""Phase 5 contract checks: the committed example sweep results validate against
the committed sweep_result.schema.json (the schema, not the TS harness, is the
public API — the same cross-language discipline as every other schema)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from conftest import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "schemas" / "sweep_result.schema.json"
SWEEP_DIR = REPO_ROOT / "data" / "examples" / "sweep"

EXPECTED_FILES = ["example-broken.json", "example-web-a.json", "example-web-b.json"]


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_sweep_schema_is_valid_2020_12() -> None:
    _validator()


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_committed_example_sweep_results_validate(name: str) -> None:
    doc = json.loads((SWEEP_DIR / name).read_text())
    errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    assert errors == [], [e.message for e in errors]


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_example_sweep_results_are_example_provenance_with_env(name: str) -> None:
    """Data integrity: fixture sweeps must never masquerade as real model data,
    and no record is valid without its env block (honesty rule, spec 5.1)."""
    doc = json.loads((SWEEP_DIR / name).read_text())
    assert doc["model_id"] == Path(name).stem
    assert len(doc["results"]) >= 1
    for record in doc["results"]:
        assert record["provenance"] == "example"
        assert record["env"]["machine_label"]
        assert record["env"]["litertjs_core_version"]


def test_broken_entry_recorded_as_failure_not_dropped() -> None:
    """The deliberately broken catalog entry produces a result file (crash
    isolation: the failure IS the result, the batch continues)."""
    doc = json.loads((SWEEP_DIR / "example-broken.json").read_text())
    for record in doc["results"]:
        assert record["loads"] is False
        assert record["runs"] is False
        assert record["failure_class"] is not None
        assert record["latency_p50_ms"] is None
