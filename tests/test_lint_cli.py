"""edge-lint CLI: golden reports, schema validation, exit-code contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from conftest import REPO_ROOT
from helpers import entry, make_doc
from litert_compat.lint.cli import app
from litert_compat.lint.report import load_lint_report_schema
from litert_compat.matrix.canonical import write_canonical

runner = CliRunner()

MATRIX = "data/examples/matrix_example.json"
FIXTURE_NAMES = {
    "clean": "data/examples/model_clean_example.tflite",
    "split": "data/examples/model_split_example.tflite",
    "mixed": "data/examples/model_mixed_example.tflite",
}
FORMAT_FLAGS = {"txt": [], "json": ["--json"], "md": ["--md"]}


@pytest.fixture(autouse=True)
def repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Goldens embed the relative paths passed on the command line."""
    monkeypatch.chdir(REPO_ROOT)


def invoke(*args: str):
    return runner.invoke(app, list(args))


@pytest.mark.parametrize("fixture", sorted(FIXTURE_NAMES))
@pytest.mark.parametrize("fmt", sorted(FORMAT_FLAGS))
def test_golden_reports(fixture: str, fmt: str) -> None:
    result = invoke(FIXTURE_NAMES[fixture], "--matrix", MATRIX, *FORMAT_FLAGS[fmt])
    assert result.exit_code == 0, result.output
    golden = (REPO_ROOT / "tests" / "golden" / f"{fixture}.{fmt}").read_text(encoding="utf-8")
    assert result.output == golden


@pytest.mark.parametrize("fixture", sorted(FIXTURE_NAMES))
def test_json_validates_against_committed_schema(fixture: str) -> None:
    result = invoke(FIXTURE_NAMES[fixture], "--matrix", MATRIX, "--json")
    assert result.exit_code == 0
    report = json.loads(result.output)
    errors = list(Draft202012Validator(load_lint_report_schema()).iter_errors(report))
    assert errors == []


def test_output_deterministic_across_runs() -> None:
    runs = [invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--json").output for _ in range(2)]
    assert runs[0] == runs[1]


def test_report_carries_backend_and_version_everywhere() -> None:
    report = json.loads(invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--json").output)
    assert report["backend"] == "gpu_mldrift"
    assert report["litert_version"] == "0.0.0-example"
    assert report["matrix"]["litert_version"] == "0.0.0-example"
    for probe in report["needs_probe"]:
        assert probe["backend"] == "gpu_mldrift"


def test_needs_probe_signatures_are_complete_and_deduped() -> None:
    report = json.loads(invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--json").output)
    probes = report["needs_probe"]
    assert [p["op"] for p in probes] == ["RESHAPE", "SOFTMAX"]
    for probe in probes:
        # complete lookup signature: enough to build a probe without the model
        assert probe["dtypes"] and "dynamic_shape" in probe["shape_meta"]
    # the mixed fixture's custom op is unknown but NOT probeable
    mixed = json.loads(invoke(FIXTURE_NAMES["mixed"], "--matrix", MATRIX, "--json").output)
    assert mixed["needs_probe"] == []
    assert mixed["custom_ops"] == [
        {"custom_code": "ExampleCustomOp", "node_index": 1, "subgraph": 0}
    ]


def test_rewrite_hints_surfaced_verbatim() -> None:
    report = json.loads(invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--json").output)
    matrix_doc = json.loads((REPO_ROOT / MATRIX).read_text(encoding="utf-8"))
    gather_entry = next(
        e for e in matrix_doc["entries"] if e["op"] == "GATHER_ND" and "rewrite_hints" in e
    )
    assert report["rewrite_suggestions"] == [
        {"subgraph": 0, "node_index": 1, "op": "GATHER_ND", "hints": gather_entry["rewrite_hints"]}
    ]


def test_findings_distinguish_matched_provenance() -> None:
    report = json.loads(invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--json").output)
    by_op = {f["op"]: f for f in report["findings"]}
    assert by_op["CONV_2D"]["provenance"] == "example"
    assert by_op["RESHAPE"]["provenance"] is None
    assert report["summary"]["matched_provenance_counts"] == {"example": 3, "unmatched": 2}


# --- exit-code contract -----------------------------------------------------


def test_clean_model_fail_on_fallback_exit_0() -> None:
    result = invoke(FIXTURE_NAMES["clean"], "--matrix", MATRIX, "--fail-on", "fallback")
    assert result.exit_code == 0


def test_fail_on_fallback_triggers_on_fallback_exit_1() -> None:
    result = invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--fail-on", "fallback")
    assert result.exit_code == 1


def test_fail_on_fallback_triggers_on_incorrect_too() -> None:
    """`incorrect` is claimed for partitioning but still a --fail-on hit."""
    result = invoke(FIXTURE_NAMES["mixed"], "--matrix", MATRIX, "--fail-on", "fallback")
    assert result.exit_code == 1
    assert "SOFTMAX (incorrect)" in result.output


def test_fail_on_partitions_threshold() -> None:
    over = invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--fail-on", "partitions:1")
    assert over.exit_code == 1
    at_limit = invoke(FIXTURE_NAMES["split"], "--matrix", MATRIX, "--fail-on", "partitions:2")
    assert at_limit.exit_code == 0


def test_fail_on_rules_combine() -> None:
    result = invoke(
        FIXTURE_NAMES["split"],
        "--matrix",
        MATRIX,
        "--fail-on",
        "fallback",
        "--fail-on",
        "partitions:1",
    )
    assert result.exit_code == 1


def test_invalid_fail_on_rule_exit_2() -> None:
    assert invoke(FIXTURE_NAMES["clean"], "--matrix", MATRIX, "--fail-on", "bogus").exit_code == 2
    assert (
        invoke(FIXTURE_NAMES["clean"], "--matrix", MATRIX, "--fail-on", "partitions:x").exit_code
        == 2
    )


def test_backend_match_ok_and_mismatch_exit_2() -> None:
    ok = invoke(FIXTURE_NAMES["clean"], "--matrix", MATRIX, "--backend", "gpu_mldrift")
    assert ok.exit_code == 0
    mismatch = invoke(FIXTURE_NAMES["clean"], "--matrix", MATRIX, "--backend", "cpu_xnnpack")
    assert mismatch.exit_code == 2


def test_json_and_md_mutually_exclusive_exit_2() -> None:
    assert invoke(FIXTURE_NAMES["clean"], "--matrix", MATRIX, "--json", "--md").exit_code == 2


def test_unparseable_model_exit_2(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_model.tflite"
    bogus.write_bytes(b"this is not a flatbuffer at all")
    result = invoke(str(bogus), "--matrix", MATRIX)
    assert result.exit_code == 2


def test_invalid_matrix_exit_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad_matrix.json"
    write_canonical(make_doc([entry(status="nope")]), bad)
    assert invoke(FIXTURE_NAMES["clean"], "--matrix", str(bad)).exit_code == 2


def test_contradictory_matrix_tie_exit_2(tmp_path: Path) -> None:
    """Distinct signatures that pass `matrix validate` but tie at lookup with
    contradictory statuses: a data defect, reported as exit 2 at lint time."""
    doc = make_doc(
        [
            entry(op="CONV_2D", status="delegated", dtypes=["float32"], constraints={"rank": 4}),
            entry(
                op="CONV_2D",
                status="crash",
                dtypes=["float32"],
                constraints={"dynamic_shape": False},
            ),
        ]
    )
    tie = tmp_path / "tie_matrix.json"
    write_canonical(doc, tie)
    result = invoke(FIXTURE_NAMES["clean"], "--matrix", str(tie))
    assert result.exit_code == 2
    assert "matrix data defect" in result.output


def test_missing_files_exit_2() -> None:
    assert invoke("no_such_model.tflite", "--matrix", MATRIX).exit_code == 2
    assert invoke(FIXTURE_NAMES["clean"], "--matrix", "no_such_matrix.json").exit_code == 2


# --- the "fixable by edge-fix" surface (--rules) ---------------------------

FIX_MODEL = "data/examples/model_fix_div_example.tflite"
FIX_MATRIX = "data/examples/matrix_fix_example.json"
RULES_DIR = "data/examples/transforms"


def test_rules_dir_annotates_resolvable_hints_with_exact_command() -> None:
    result = invoke(FIX_MODEL, "--matrix", FIX_MATRIX, "--rules", RULES_DIR)
    assert result.exit_code == 0  # no --fail-on: annotation changes no verdict
    assert "fixable by edge-fix rule 'example-replace-div-floor-div'" in result.output
    assert (
        f"run: edge-fix {FIX_MODEL} --matrix {FIX_MATRIX} --rules {RULES_DIR}"
        in result.output
    )


def test_rules_dir_annotates_markdown_too() -> None:
    result = invoke(FIX_MODEL, "--matrix", FIX_MATRIX, "--rules", RULES_DIR, "--md")
    assert "fixable by edge-fix rule 'example-replace-div-floor-div'" in result.output


def test_unresolvable_transform_id_is_reported(tmp_path: Path) -> None:
    empty = tmp_path / "rules"
    empty.mkdir()
    result = invoke(FIX_MODEL, "--matrix", FIX_MATRIX, "--rules", str(empty))
    assert "does not resolve in" in result.output
    assert "run: edge-fix" not in result.output


def test_rules_dir_never_changes_the_json_report() -> None:
    plain = invoke(FIX_MODEL, "--matrix", FIX_MATRIX, "--json")
    with_rules = invoke(FIX_MODEL, "--matrix", FIX_MATRIX, "--json", "--rules", RULES_DIR)
    # stderr note is allowed; the report bytes are identical (verbatim rule).
    assert plain.stdout == with_rules.stdout


def test_invalid_rules_dir_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "rules"
    bad.mkdir()
    (bad / "broken.json").write_text("{not json", encoding="utf-8")
    result = invoke(FIX_MODEL, "--matrix", FIX_MATRIX, "--rules", str(bad))
    assert result.exit_code == 2
