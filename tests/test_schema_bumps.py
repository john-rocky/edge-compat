"""Phase 6 / 10 / 13 schema bumps are additive: old documents still validate,
the `browser` and `device` card blocks are tied to their schema versions, and
the registered browser and NPU backend IDs validate as matrix snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import REPO_ROOT
from litert_compat.cards.schema_io import schema_errors
from litert_compat.matrix.validate import validate_document

EXAMPLES = REPO_ROOT / "data" / "examples"
WEB_BACKENDS = ("wasm_xnnpack", "webgpu_mldrift", "webnn")
NPU_BACKENDS = ("npu_qnn", "npu_neuropilot")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --- matrix.schema.json 1.1 ---


def test_committed_1_0_matrix_example_still_validates() -> None:
    doc = _load(EXAMPLES / "matrix_example.json")
    assert doc["schema_version"] == "1.0"
    assert validate_document(doc) == []


@pytest.mark.parametrize("backend", WEB_BACKENDS)
def test_web_backend_matrix_snapshots_validate(backend: str) -> None:
    doc = _load(EXAMPLES / f"matrix_{backend}_example.json")
    assert doc["schema_version"] == "1.1"
    assert doc["backend"] == backend
    assert doc["litert_version"] == "2.5.3"  # the version axis is @litertjs/core
    assert validate_document(doc) == []


@pytest.mark.parametrize("backend", WEB_BACKENDS)
def test_web_backend_snapshots_carry_no_entries(backend: str) -> None:
    """The trap rule: registration adds backend IDs, never op-level entries."""
    doc = _load(EXAMPLES / f"matrix_{backend}_example.json")
    assert doc["entries"] == []


def test_matrix_schema_rejects_unknown_version() -> None:
    doc = _load(EXAMPLES / "matrix_example.json")
    doc["schema_version"] = "9.9"
    assert validate_document(doc) != []


# --- card.schema.json 1.1 / 1.2 ---


def test_1_0_card_shape_still_validates() -> None:
    """Additivity: a document without the browser/device blocks validates as
    1.0 (the committed tiny cards gained device blocks in Phase 13, so the
    1.0 shape is derived by removing the additive block)."""
    card = _load(REPO_ROOT / "cards" / "example-tiny-clean" / "card.json")
    del card["device"]
    card["schema_version"] = "1.0"
    assert schema_errors(card, "card.schema.json") == []


def test_committed_enriched_card_validates() -> None:
    card = _load(REPO_ROOT / "cards" / "example-web-a" / "card.json")
    assert card["schema_version"] == "1.1"
    assert card["browser"]["backends"]
    assert schema_errors(card, "card.schema.json") == []


def test_browser_block_requires_schema_1_1_or_1_2() -> None:
    card = _load(REPO_ROOT / "cards" / "example-web-a" / "card.json")
    card["schema_version"] = "1.0"
    assert schema_errors(card, "card.schema.json") != []
    card["schema_version"] = "1.2"  # browser is allowed on 1.2 (device-era cards)
    assert schema_errors(card, "card.schema.json") == []


def test_card_1_1_without_browser_block_is_valid() -> None:
    """Additivity both ways: 1.1 marks the browser feature available, not required."""
    card = _load(REPO_ROOT / "cards" / "example-tiny-clean" / "card.json")
    del card["device"]
    card["schema_version"] = "1.1"
    assert schema_errors(card, "card.schema.json") == []


# --- matrix.schema.json 1.3 + card.schema.json 1.2 (Phase 13) ---


@pytest.mark.parametrize("backend", NPU_BACKENDS)
def test_npu_backend_matrix_snapshots_validate(backend: str) -> None:
    doc = _load(EXAMPLES / f"matrix_{backend}_example.json")
    assert doc["schema_version"] == "1.3"
    assert doc["backend"] == backend
    assert doc["litert_version"] == "2.1.6"  # the version axis is native litert
    assert validate_document(doc) == []


@pytest.mark.parametrize("backend", NPU_BACKENDS)
def test_npu_backend_snapshots_carry_no_entries(backend: str) -> None:
    """The trap rule, device edition: registration adds backend IDs, never
    op-level entries."""
    doc = _load(EXAMPLES / f"matrix_{backend}_example.json")
    assert doc["entries"] == []


def test_npu_snapshot_requires_target() -> None:
    """NPU behavior varies per SoC and vendor SDK: an NPU snapshot without a
    full target (device, soc, driver) fails semantic lint."""
    doc = _load(EXAMPLES / "matrix_npu_qnn_example.json")
    doc.pop("target")
    assert any(f.code == "npu_target_required" for f in validate_document(doc))
    doc = _load(EXAMPLES / "matrix_npu_qnn_example.json")
    doc["target"].pop("driver")
    assert any(f.code == "npu_target_required" for f in validate_document(doc))
    # Non-NPU backends are unaffected: stripping target stays valid there.
    base = _load(EXAMPLES / "matrix_example.json")
    base.pop("target", None)
    assert validate_document(base) == []


def test_committed_device_enriched_cards_validate() -> None:
    for model_id in ("example-tiny-clean", "example-tiny-mixed"):
        card = _load(REPO_ROOT / "cards" / model_id / "card.json")
        assert card["schema_version"] == "1.2"
        assert card["device"]["records"]
        assert schema_errors(card, "card.schema.json") == []


def test_device_block_requires_schema_1_2() -> None:
    card = _load(REPO_ROOT / "cards" / "example-tiny-clean" / "card.json")
    card["schema_version"] = "1.1"
    assert schema_errors(card, "card.schema.json") != []


def test_card_1_2_without_device_block_is_valid() -> None:
    """Additivity both ways: 1.2 marks the device feature available, not required."""
    card = _load(REPO_ROOT / "cards" / "example-web-a" / "card.json")
    assert "device" not in card
    card["schema_version"] = "1.2"
    assert schema_errors(card, "card.schema.json") == []


def test_card_device_run_shape_is_the_device_run_schema_shape() -> None:
    """The accelerator record exists once (device_run_result.schema.json
    $defs); the card embeds it verbatim — a snapshot record is valid inside a
    card unchanged."""
    snapshot = _load(
        EXAMPLES
        / "device_runs" / "0.0.0-example" / "2026-01-01"
        / "example-tiny-clean__example-phone.json"
    )
    card = _load(REPO_ROOT / "cards" / "example-tiny-clean" / "card.json")
    assert [e["run"] for e in card["device"]["records"]] == snapshot["results"]


def test_card_browser_env_shape_is_the_sweep_env_shape() -> None:
    """The env block exists once (sweep_result.schema.json $defs/env); the card
    schema references it — a sweep env is valid verbatim inside a card."""
    sweep = _load(EXAMPLES / "sweep" / "example-web-a.json")
    card = _load(REPO_ROOT / "cards" / "example-web-a" / "card.json")
    assert [r["env"] for r in card["browser"]["backends"]] == [
        r["env"] for r in sweep["results"]
    ]


# --- matrix.schema.json 1.2 + lint_report.schema.json 1.1 (Phase 10) ---


def test_fix_demo_matrix_validates_as_1_2() -> None:
    doc = _load(EXAMPLES / "matrix_fix_example.json")
    assert doc["schema_version"] == "1.2"
    assert validate_document(doc) == []


def test_matrix_1_2_hints_carry_transform_ids() -> None:
    """Every transform_id in the demo matrix names a committed example rule."""
    doc = _load(EXAMPLES / "matrix_fix_example.json")
    transform_ids = {
        hint["transform_id"]
        for entry in doc["entries"]
        for hint in entry.get("rewrite_hints", [])
        if "transform_id" in hint
    }
    assert transform_ids  # the link exists
    for transform_id in transform_ids:
        rule_path = EXAMPLES / "transforms" / f"{transform_id}.json"
        assert rule_path.is_file(), transform_id
        assert _load(rule_path)["id"] == transform_id


def test_matrix_1_0_rejects_transform_id_free_ride() -> None:
    """transform_id rides on schema 1.2 documents; committed 1.0/1.1 examples
    carry none (proven by scanning them)."""
    for name in ("matrix_example.json", "matrix_webgpu_mldrift_example.json"):
        doc = _load(EXAMPLES / name)
        for entry in doc["entries"]:
            for hint in entry.get("rewrite_hints", []):
                assert "transform_id" not in hint


def test_lint_passes_transform_id_through_verbatim() -> None:
    """A lint run against a 1.2 matrix surfaces the hint's transform_id
    unchanged, and the 1.1 report still validates against the lint schema."""
    from jsonschema import Draft202012Validator
    from typer.testing import CliRunner

    from litert_compat.lint.cli import app as lint_app
    from litert_compat.lint.report import load_lint_report_schema

    result = CliRunner().invoke(
        lint_app,
        [
            str(EXAMPLES / "model_fix_square_example.tflite"),
            "--matrix", str(EXAMPLES / "matrix_fix_example.json"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["schema_version"] == "1.1"
    hints = [h for s in report["rewrite_suggestions"] for h in s["hints"]]
    assert [h.get("transform_id") for h in hints] == ["example-decompose-square-mul"]
    errors = list(Draft202012Validator(load_lint_report_schema()).iter_errors(report))
    assert errors == []


def test_committed_1_0_lint_examples_still_validate() -> None:
    """Additivity: the lint schema accepts both 1.0 and 1.1 reports (the
    committed examples were regenerated to 1.1; a 1.0 shape stays valid)."""
    from jsonschema import Draft202012Validator

    from litert_compat.lint.report import load_lint_report_schema

    report = _load(EXAMPLES / "lint_clean_example.json")
    assert report["schema_version"] == "1.1"
    validator = Draft202012Validator(load_lint_report_schema())
    assert list(validator.iter_errors(report)) == []
    report["schema_version"] = "1.0"
    assert list(validator.iter_errors(report)) == []
    report["schema_version"] = "9.9"
    assert list(validator.iter_errors(report)) != []


# --- transform_rule / fix_report schemas are registered ---


def test_new_phase_10_schemas_resolve_cross_file_refs() -> None:
    """fix_report $refs the lint summary shape (it exists exactly once)."""
    minimal_summary = _load(EXAMPLES / "lint_clean_example.json")["summary"]
    assert schema_errors(
        {"summary": minimal_summary}, "fix_report.schema.json"
    ) != []  # not a full report — but the $ref itself must resolve, not crash
