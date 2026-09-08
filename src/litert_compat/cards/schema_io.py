"""Schema loading and validation for the cards pipeline.

`card.schema.json` references `benchmark_result.schema.json` (the benchmark
record shape exists exactly once) and, since schema 1.1, the sweep env block
in `sweep_result.schema.json` (the env shape exists exactly once too); a
`referencing.Registry` resolves the cross-file `$ref`s from the committed
schema files themselves.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

_SCHEMA_FILES = (
    "card.schema.json",
    "benchmark_result.schema.json",
    "lint_report.schema.json",
    "sweep_result.schema.json",
    "transform_rule.schema.json",
    "fix_report.schema.json",
    "device_run_result.schema.json",
)


def load_schema(name: str) -> dict[str, Any]:
    """Load a committed schema — packaged copy first, repo root when running from source."""
    res = resources.files("litert_compat").joinpath(f"schemas/{name}")
    if res.is_file():
        return json.loads(res.read_text(encoding="utf-8"))
    repo_copy = Path(__file__).resolve().parents[3] / "schemas" / name
    return json.loads(repo_copy.read_text(encoding="utf-8"))


def _validator(name: str) -> Draft202012Validator:
    registry: Registry[Any] = Registry()
    for schema_name in _SCHEMA_FILES:
        schema = load_schema(schema_name)
        registry = registry.with_resource(schema_name, Resource.from_contents(schema))
    return Draft202012Validator(load_schema(name), registry=registry)


def schema_errors(doc: Any, schema_name: str) -> list[str]:
    """Validation errors of `doc` against a committed schema; empty list = valid."""
    validator = _validator(schema_name)
    errors = sorted(validator.iter_errors(doc), key=lambda e: (list(e.absolute_path), e.message))
    out = []
    for err in errors:
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        out.append(f"{where}: {err.message}")
    return out
