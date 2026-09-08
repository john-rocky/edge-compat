"""`edge-fix rules reverify` (Phase 11.4): re-run every rule's declared
fixture; failing rules are flagged `stale` in place (never auto-deleted),
passing rules have the flag cleared. Plus the transform_rule 1.0 -> 1.1
additive schema bump.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.cards.schema_io import schema_errors
from litert_compat.fix.cli import app as fix_app
from litert_compat.fix.rules import load_rules
from litert_compat.matrix.canonical import canonical_dumps
from litert_compat.probe.runners import CpuRunner

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
EXAMPLE_RULES_DIR = EXAMPLES / "transforms"
RUNNERS_AVAILABLE = CpuRunner().availability() is None


def _copy_example_rules(tmp_path: Path) -> Path:
    """Example rules + the fixtures they declare, in a writable directory."""
    rules_dir = tmp_path / "transforms"
    shutil.copytree(EXAMPLE_RULES_DIR, rules_dir)
    for name in (
        "model_fix_div_example.tflite",
        "model_fix_square_example.tflite",
        "model_fix_int_mul_example.tflite",
        "model_fix_io64_example.tflite",
        "matrix_fix_example.json",
    ):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    return rules_dir


def _write_unmatchable_rule(rules_dir: Path) -> Path:
    """A schema-valid rule whose fixture dry-run cannot succeed: it matches an
    op the fixture model does not contain, so the outcome is no_rules_matched
    — a static, runner-independent failure."""
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "id": "test-unmatchable",
        "title": "matches nothing in its own fixture",
        "match": {"op": "SUB", "dtypes": ["int32"]},
        "action": "replace_op",
        "params": {"new_op": "ADD"},
        "expected_effect": "never observed",
        "provenance": "example",
        "fixture": {
            "model": "../model_fix_div_example.tflite",
            "matrix": "../matrix_fix_example.json",
        },
    }
    path = rules_dir / "test-unmatchable.json"
    path.write_text(canonical_dumps(doc), encoding="utf-8")
    return path


# --- schema bump ------------------------------------------------------------


def test_committed_example_rules_still_validate_after_bump() -> None:
    """Additivity: the committed 1.0 rule documents pass the 1.1 schema."""
    for path in sorted(EXAMPLE_RULES_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["schema_version"] == "1.0"
        assert schema_errors(doc, "transform_rule.schema.json") == [], path.name


def _stale_block() -> dict[str, str]:
    return {
        "date": "2026-08-10",
        "runtime": "ai-edge-litert",
        "runtime_version": "9.9.9",
        "reason": "fixture dry-run outcome: no_improvement",
    }


def test_stale_block_requires_1_1() -> None:
    base = json.loads(
        (EXAMPLE_RULES_DIR / "example-replace-div-floor-div.json").read_text(encoding="utf-8")
    )
    stale = {**base, "stale": _stale_block(), "schema_version": "1.1"}
    assert schema_errors(stale, "transform_rule.schema.json") == []
    downgraded = {**stale, "schema_version": "1.0"}
    assert schema_errors(downgraded, "transform_rule.schema.json") != []
    unknown = {**base, "schema_version": "2.0"}
    assert schema_errors(unknown, "transform_rule.schema.json") != []
    incomplete = {**stale, "stale": {"date": "2026-08-10"}}
    assert schema_errors(incomplete, "transform_rule.schema.json") != []


# --- reverify ---------------------------------------------------------------


@pytest.mark.skipif(not RUNNERS_AVAILABLE, reason="edge-compat[runners] not installed")
def test_reverify_passing_rules_exit_0_and_touch_nothing(tmp_path: Path) -> None:
    rules_dir = _copy_example_rules(tmp_path)
    before = {p.name: p.read_bytes() for p in sorted(rules_dir.glob("*.json"))}
    result = runner.invoke(
        fix_app, ["rules", "reverify", str(rules_dir), "--date", "2026-08-10", "--json"]
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert [r["status"] for r in rows] == ["pass"] * 4
    assert {p.name: p.read_bytes() for p in sorted(rules_dir.glob("*.json"))} == before


def test_reverify_flags_failing_rule_stale_and_clears_on_pass(tmp_path: Path) -> None:
    rules_dir = tmp_path / "transforms"
    rules_dir.mkdir()
    shutil.copy(EXAMPLES / "model_fix_div_example.tflite", tmp_path)
    shutil.copy(EXAMPLES / "matrix_fix_example.json", tmp_path)
    rule_path = _write_unmatchable_rule(rules_dir)

    args = ["rules", "reverify", str(rules_dir), "--date", "2026-08-10",
            "--runtime-version", "0.0.0-test"]
    result = runner.invoke(fix_app, args)
    assert result.exit_code == 1, result.output

    doc = json.loads(rule_path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.1"
    assert doc["stale"]["runtime_version"] == "0.0.0-test"
    assert "no_rules_matched" in doc["stale"]["reason"]
    assert schema_errors(doc, "transform_rule.schema.json") == []
    # The flagged rule still loads: stale rules are surfaced, not disabled.
    assert load_rules(rules_dir)[0].stale is not None

    # Idempotent: a second run changes nothing.
    bytes_after_first = rule_path.read_bytes()
    result = runner.invoke(fix_app, args)
    assert result.exit_code == 1
    assert rule_path.read_bytes() == bytes_after_first

    # Repair the rule (match the fixture's real finding); reverify clears the
    # flag — the same upgrade-on-re-measurement pattern as inferred->measured.
    good = json.loads(
        (EXAMPLE_RULES_DIR / "example-replace-div-floor-div.json").read_text(encoding="utf-8")
    )
    doc.update(match=good["match"], params=good["params"])
    rule_path.write_text(canonical_dumps(doc), encoding="utf-8")
    if not RUNNERS_AVAILABLE:
        pytest.skip("clearing requires a passing verification (runners extra)")
    result = runner.invoke(fix_app, args)
    assert result.exit_code == 0, result.output
    cleared = json.loads(rule_path.read_text(encoding="utf-8"))
    assert "stale" not in cleared
    assert cleared["schema_version"] == "1.1"
    assert schema_errors(cleared, "transform_rule.schema.json") == []


def test_reverify_without_interpreter_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the measuring instrument (and no --runtime-version), reverify
    refuses: no measurement, no stale flag — data integrity rule 1."""
    import litert_compat.fix.cli as fix_cli

    def missing(_name: str) -> str:
        raise fix_cli.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(fix_cli.importlib.metadata, "version", missing)
    rules_dir = _copy_example_rules(tmp_path)
    result = runner.invoke(fix_app, ["rules", "reverify", str(rules_dir)])
    assert result.exit_code == 2
    assert "no rule is ever flagged stale" in result.stderr


def test_reverify_performs_zero_matrix_writes(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "data" / "matrix"
    matrix_dir.mkdir(parents=True)
    shutil.copy(
        EXAMPLES / "matrix_example.json", matrix_dir / "gpu_mldrift__0.0.0-example.json"
    )
    rules_dir = tmp_path / "transforms"
    rules_dir.mkdir()
    shutil.copy(EXAMPLES / "model_fix_div_example.tflite", tmp_path)
    shutil.copy(EXAMPLES / "matrix_fix_example.json", tmp_path)
    _write_unmatchable_rule(rules_dir)
    before = {p.name: p.read_bytes() for p in sorted(matrix_dir.iterdir())}
    result = runner.invoke(
        fix_app,
        ["rules", "reverify", str(rules_dir), "--date", "2026-08-10",
         "--runtime-version", "0.0.0-test"],
    )
    assert result.exit_code == 1
    assert {p.name: p.read_bytes() for p in sorted(matrix_dir.iterdir())} == before
