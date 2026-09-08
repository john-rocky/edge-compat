"""Transform rule loading: schema + semantic validation, strict directory load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import REPO_ROOT
from litert_compat.fix.rules import RuleError, load_rule_file, load_rules, scaffold_rule

EXAMPLE_RULES_DIR = REPO_ROOT / "data" / "examples" / "transforms"
EXPECTED_EXAMPLE_IDS = [
    "example-decompose-square-mul",
    "example-insert-cast-mul-int32",
    "example-io-cast-int64",
    "example-replace-div-floor-div",
]


def valid_rule(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "id": "test-rule",
        "title": "test rule",
        "match": {"op": "DIV", "dtypes": ["int32"]},
        "action": "replace_op",
        "params": {"new_op": "FLOOR_DIV"},
        "expected_effect": "delegates",
        "provenance": "example",
        "fixture": {"model": "m.tflite", "matrix": "x.json"},
    }
    doc.update(overrides)
    return doc


def write_rule(tmp_path: Path, doc: dict[str, Any], name: str | None = None) -> Path:
    path = tmp_path / f"{name or doc['id']}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_example_rules_load_sorted_by_id() -> None:
    rules = load_rules(EXAMPLE_RULES_DIR)
    assert [r.id for r in rules] == EXPECTED_EXAMPLE_IDS
    assert all(r.provenance == "example" for r in rules)
    # Every v1 action is covered by exactly one example rule.
    assert sorted(r.action for r in rules) == [
        "decompose", "insert_cast", "io_cast", "replace_op",
    ]


def test_example_rule_fixtures_exist() -> None:
    for rule in load_rules(EXAMPLE_RULES_DIR):
        assert rule.fixture_model_path().is_file(), rule.id
        assert rule.fixture_matrix_path().is_file(), rule.id


def test_valid_rule_loads(tmp_path: Path) -> None:
    rule = load_rule_file(write_rule(tmp_path, valid_rule()))
    assert rule.id == "test-rule"
    assert not rule.uses_op_sequence


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"action": "delete_node"}, "delete_node"),  # closed vocabulary
        ({"fixture": None}, "fixture"),  # declared fixture is required
        ({"match": {"dtypes": ["int32"]}}, "match"),  # node action needs op
        ({"match": {"op": "DIV", "op_sequence": ["DIV", "MUL"], "dtypes": ["int32"]}}, "match"),
        ({"params": {}}, "new_op"),  # replace_op needs new_op
        ({"id": "Bad_ID"}, "id"),  # kebab-case ids only
    ],
)
def test_schema_rejections(tmp_path: Path, mutation: dict[str, Any], match: str) -> None:
    doc = valid_rule(**mutation)
    if mutation.get("fixture", "keep") is None:
        del doc["fixture"]
    with pytest.raises(RuleError, match=match):
        load_rule_file(write_rule(tmp_path, doc, name=doc.get("id", "test-rule")))


def test_io_cast_match_with_op_is_rejected(tmp_path: Path) -> None:
    doc = valid_rule(
        action="io_cast",
        match={"op": "DIV", "dtypes": ["int64"]},
        params={"to": "int32"},
    )
    with pytest.raises(RuleError):
        load_rule_file(write_rule(tmp_path, doc))


def test_replace_op_outside_options_registry_is_rejected(tmp_path: Path) -> None:
    # CONV_2D options (strides) are per-instance — no registry default exists.
    doc = valid_rule(params={"new_op": "CONV_2D"})
    with pytest.raises(RuleError, match="options registry"):
        load_rule_file(write_rule(tmp_path, doc))


def test_concatenation_emission_is_rejected(tmp_path: Path) -> None:
    # CONCATENATION's axis is load-bearing per instance — never defaulted.
    doc = valid_rule(params={"new_op": "CONCATENATION"})
    with pytest.raises(RuleError, match="options registry"):
        load_rule_file(write_rule(tmp_path, doc))


@pytest.mark.parametrize(
    "new_op",
    ["SELECT", "SELECT_V2", "NOT_EQUAL", "MEAN", "PAD", "TRANSPOSE",
     "FULLY_CONNECTED", "BATCH_MATMUL", "GELU", "HARD_SWISH", "SOFTMAX",
     "RESIZE_BILINEAR"],
)
def test_empty_table_families_are_emittable(new_op: str, tmp_path: Path) -> None:
    """The 2026-08-12 emission extension: ops whose canonical options table is
    complete and valid on its own (plus SOFTMAX beta=1.0) load as rule targets."""
    doc = valid_rule(params={"new_op": new_op})
    rule = load_rule_file(write_rule(tmp_path, doc))
    assert rule.params["new_op"] == new_op


def test_emitted_options_are_canonical() -> None:
    from litert_compat.fix.options_registry import default_options_for

    softmax = default_options_for("SOFTMAX")
    assert softmax is not None and softmax.type_code == 9
    assert softmax.fields[0].value == 1.0
    mean = default_options_for("MEAN")
    assert mean is not None and mean.type_code == 27 and mean.fields == ()
    gelu = default_options_for("GELU")
    assert gelu is not None and gelu.type_code == 116 and gelu.fields == ()


def test_every_emittable_type_code_is_loadable() -> None:
    """A fixed model must reload for a second pass: every union type the
    emission registry can write is registered in the loading layouts."""
    from litert_compat.fix.options_registry import (
        _EMPTY_TABLE_EMISSION,
        OPTION_LAYOUTS,
    )

    assert set(_EMPTY_TABLE_EMISSION.values()) <= set(OPTION_LAYOUTS)


@pytest.mark.parametrize(
    ("nodes", "intermediates", "match"),
    [
        # tmp consumed before produced
        (
            [
                {"op": "MUL", "inputs": ["tmp:t", "in:0"], "outputs": ["out:0"]},
                {"op": "NEG", "inputs": ["in:0"], "outputs": ["tmp:t"]},
            ],
            [{"name": "t", "dtype": "like:in:0", "shape": "like:in:0"}],
            "consumed before",
        ),
        # out written twice
        (
            [
                {"op": "NEG", "inputs": ["in:0"], "outputs": ["out:0"]},
                {"op": "NEG", "inputs": ["in:0"], "outputs": ["out:0"]},
            ],
            None,
            "more than once",
        ),
        # non-contiguous outputs
        ([{"op": "NEG", "inputs": ["in:0"], "outputs": ["out:1"]}], None, "contiguously"),
        # undeclared intermediate
        ([{"op": "NEG", "inputs": ["in:0"], "outputs": ["tmp:ghost"]}], None, "declared"),
        # unused intermediate
        (
            [{"op": "NEG", "inputs": ["in:0"], "outputs": ["out:0"]}],
            [{"name": "unused", "dtype": "float32", "shape": "like:in:0"}],
            "never used",
        ),
    ],
)
def test_decompose_recipe_rejections(
    tmp_path: Path, nodes: list[dict[str, Any]], intermediates: Any, match: str
) -> None:
    params: dict[str, Any] = {"nodes": nodes}
    if intermediates is not None:
        params["intermediates"] = intermediates
    doc = valid_rule(action="decompose", match={"op": "SQUARE"}, params=params)
    with pytest.raises(RuleError, match=match):
        load_rule_file(write_rule(tmp_path, doc))


def test_measured_rule_requires_evidence(tmp_path: Path) -> None:
    doc = valid_rule(provenance="measured")
    with pytest.raises(RuleError, match="evidence"):
        load_rule_file(write_rule(tmp_path, doc))


def test_filename_must_match_id(tmp_path: Path) -> None:
    with pytest.raises(RuleError, match=r"named <id>\.json"):
        load_rule_file(write_rule(tmp_path, valid_rule(), name="other-name"))


def test_duplicate_ids_across_directory(tmp_path: Path) -> None:
    write_rule(tmp_path, valid_rule())
    # Same id, different file name: the file itself fails the name check AND
    # the directory load reports every finding.
    write_rule(tmp_path, valid_rule(), name="test-rule-copy")
    with pytest.raises(RuleError):
        load_rules(tmp_path)


def test_missing_rules_directory(tmp_path: Path) -> None:
    with pytest.raises(RuleError, match="not found"):
        load_rules(tmp_path / "nope")


def test_op_sequence_rule_loads_but_is_flagged(tmp_path: Path) -> None:
    doc = valid_rule(match={"op_sequence": ["DIV", "MUL"], "dtypes": ["int32"]})
    rule = load_rule_file(write_rule(tmp_path, doc))
    assert rule.uses_op_sequence


@pytest.mark.parametrize("action", ["replace_op", "decompose", "insert_cast", "io_cast"])
def test_scaffold_is_schema_valid(action: str) -> None:
    """The scaffold passes the JSON Schema as written; the TODO fields fail
    only the semantic/fixture checks the author resolves next."""
    from litert_compat.cards.schema_io import schema_errors

    doc = scaffold_rule(
        rule_id="my-rule", action=action, op="DIV",
        fixture_model="m.tflite", fixture_matrix="x.json", generated_at="2026-08-10",
    )
    assert schema_errors(doc, "transform_rule.schema.json") == []
