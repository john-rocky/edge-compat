"""edge-fix CLI: default-command dispatch, exit codes, report output, and
the rule authoring path (rules new → fill TODOs → rules validate)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.cards.schema_io import schema_errors
from litert_compat.fix.cli import app

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
MODEL = str(EXAMPLES / "model_fix_div_example.tflite")
MATRIX = str(EXAMPLES / "matrix_fix_example.json")
RULES = str(EXAMPLES / "transforms")


def invoke(*args: str):
    return runner.invoke(app, list(args))


def test_bare_model_argument_dispatches_to_run() -> None:
    """`edge-fix MODEL.tflite ...` — the spec's CLI shape."""
    result = invoke(MODEL, "--matrix", MATRIX, "--rules", RULES)
    assert result.exit_code == 0, result.output
    assert "edge-fix: fixed" in result.output


def test_explicit_run_subcommand_is_equivalent() -> None:
    bare = invoke(MODEL, "--matrix", MATRIX, "--rules", RULES)
    explicit = invoke("run", MODEL, "--matrix", MATRIX, "--rules", RULES)
    assert bare.output == explicit.output


def test_backend_must_match_matrix() -> None:
    result = invoke(MODEL, "--matrix", MATRIX, "--rules", RULES, "--backend", "webnn")
    assert result.exit_code == 2


def test_apply_requires_out(tmp_path: Path) -> None:
    assert invoke(MODEL, "--matrix", MATRIX, "--rules", RULES, "--apply").exit_code == 2
    assert invoke(
        MODEL, "--matrix", MATRIX, "--rules", RULES,
        "--apply", "--dry-run", "--out", str(tmp_path / "f.tflite"),
    ).exit_code == 2
    assert invoke(
        MODEL, "--matrix", MATRIX, "--rules", RULES, "--out", str(tmp_path / "f.tflite")
    ).exit_code == 2


def test_never_in_place() -> None:
    result = invoke(MODEL, "--matrix", MATRIX, "--rules", RULES, "--apply", "--out", MODEL)
    assert result.exit_code == 2
    assert "never modified in place" in result.output


def test_unparseable_model_is_usage_error() -> None:
    result = invoke(
        str(EXAMPLES / "model_broken_example.tflite"), "--matrix", MATRIX, "--rules", RULES
    )
    assert result.exit_code == 2


def test_invalid_rules_dir_is_usage_error(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    result = invoke(MODEL, "--matrix", MATRIX, "--rules", str(tmp_path))
    assert result.exit_code == 2
    assert "bad.json" in result.output


def test_no_rules_matched_exits_1(tmp_path: Path) -> None:
    empty_rules = tmp_path / "rules"
    empty_rules.mkdir()
    result = invoke(MODEL, "--matrix", MATRIX, "--rules", str(empty_rules))
    assert result.exit_code == 1
    assert "no_rules_matched" in result.output
    assert "unmatched" in result.output


def test_json_report_is_written_and_valid(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    result = invoke(
        MODEL, "--matrix", MATRIX, "--rules", RULES, "--json", str(report_path)
    )
    assert result.exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert schema_errors(report, "fix_report.schema.json") == []
    assert report["mode"] == "dry_run"
    assert report["outcome"] == "fixed"


def test_apply_writes_fixed_model(tmp_path: Path) -> None:
    out = tmp_path / "fixed.tflite"
    report_path = tmp_path / "report.json"
    result = invoke(
        MODEL, "--matrix", MATRIX, "--rules", RULES,
        "--apply", "--out", str(out), "--json", str(report_path),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["unverified"]:  # runners extra absent: --apply must refuse
        assert result.exit_code == 2
        assert not out.exists()
    else:
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert report["output"]["written"] is True
    assert schema_errors(report, "fix_report.schema.json") == []


def test_cli_output_deterministic_across_runs(tmp_path: Path) -> None:
    reports = []
    for i in range(2):
        report_path = tmp_path / f"r{i}.json"
        result = invoke(MODEL, "--matrix", MATRIX, "--rules", RULES,
                        "--json", str(report_path))
        assert result.exit_code == 0
        reports.append(report_path.read_bytes())
    assert reports[0] == reports[1]


# --- rule authoring path -----------------------------------------------------


def test_rules_new_then_fill_then_validate(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    scaffolded = invoke(
        "rules", "new", "--id", "my-div-rule", "--action", "replace_op", "--op", "DIV",
        "--out-dir", str(rules_dir), "--generated-at", "2026-08-10",
    )
    assert scaffolded.exit_code == 0, scaffolded.output
    path = rules_dir / "my-div-rule.json"
    assert path.exists()

    # The scaffold's TODO fields fail validation with precise reasons.
    todo_result = invoke("rules", "validate", str(path))
    assert todo_result.exit_code == 1
    assert "TODO" in todo_result.output

    # Fill the TODOs the way an author would.
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["title"] = "replace DIV with FLOOR_DIV"
    doc["match"]["dtypes"] = ["int32"]
    doc["params"]["new_op"] = "FLOOR_DIV"
    doc["expected_effect"] = "DIV delegates as FLOOR_DIV"
    doc["notes"] = "non-negative operands only"
    doc["evidence"] = {
        "source_model": "example-synthetic",
        "litert_version": "0.0.0-example",
        "date": "2026-08-10",
    }
    doc["fixture"] = {
        "model": os.path.relpath(MODEL, rules_dir),
        "matrix": os.path.relpath(MATRIX, rules_dir),
    }
    path.write_text(json.dumps(doc), encoding="utf-8")

    result = invoke("rules", "validate", str(path))
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_rules_new_refuses_overwrite(tmp_path: Path) -> None:
    args = ["rules", "new", "--id", "x-rule", "--action", "insert_cast",
            "--out-dir", str(tmp_path)]
    assert invoke(*args).exit_code == 0
    assert invoke(*args).exit_code == 2


def test_rules_validate_flags_non_improving_rule(tmp_path: Path) -> None:
    # Valid schema, applies cleanly, but MUL[int32] is fallback too: the
    # fixture dry-run must FAIL with the no_improvement outcome.
    doc = {
        "schema_version": "1.0",
        "id": "div-to-mul",
        "title": "does not improve",
        "match": {"op": "DIV", "dtypes": ["int32"]},
        "action": "replace_op",
        "params": {"new_op": "MUL"},
        "expected_effect": "nothing",
        "provenance": "example",
        "fixture": {
            "model": os.path.relpath(MODEL, tmp_path),
            "matrix": os.path.relpath(MATRIX, tmp_path),
        },
    }
    path = tmp_path / "div-to-mul.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = invoke("rules", "validate", str(path))
    assert result.exit_code == 1
    assert "no_improvement" in result.output


def test_example_rules_validate_end_to_end() -> None:
    """The committed example rules pass `rules validate` (DoD: every v1 action
    passes end-to-end on synthetic fixtures)."""
    paths = sorted((EXAMPLES / "transforms").glob("*.json"))
    assert len(paths) == 4
    result = invoke("rules", "validate", *[str(p) for p in paths])
    assert result.exit_code == 0, result.output
    assert result.output.count("PASS") == 4


@pytest.mark.parametrize("action", ["replace_op", "decompose", "insert_cast", "io_cast"])
def test_rules_new_supports_every_action(action: str, tmp_path: Path) -> None:
    result = invoke("rules", "new", "--id", f"r-{action.replace('_', '-')}",
                    "--action", action, "--out-dir", str(tmp_path))
    assert result.exit_code == 0, result.output
