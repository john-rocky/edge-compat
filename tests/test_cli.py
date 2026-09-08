from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from helpers import entry, make_doc
from litert_compat.cli import app
from litert_compat.matrix import validate_document
from litert_compat.matrix.canonical import canonical_dumps, load_json, write_canonical

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def write_doc(path: Path, doc: dict) -> Path:
    write_canonical(doc, path)
    return path


def measured_doc() -> dict:
    return make_doc(
        [
            entry(
                provenance="measured",
                evidence={
                    "source_model": "model-a",
                    "litert_version": "1.2.0",
                    "date": "2026-08-01",
                },
            )
        ],
        litert_version="1.2.0",
        generated_at="2026-08-01",
    )


def test_validate_pass_exit_0(example_matrix_path: Path) -> None:
    result = invoke("matrix", "validate", str(example_matrix_path))
    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "example=11" in result.output


def test_validate_findings_exit_1(tmp_path: Path) -> None:
    path = write_doc(tmp_path / "bad.json", make_doc([entry(provenance="measured")]))
    result = invoke("matrix", "validate", str(path))
    assert result.exit_code == 1
    assert "missing_evidence" in result.output


def test_validate_missing_file_exit_2() -> None:
    assert invoke("matrix", "validate", "does_not_exist.json").exit_code == 2


def test_validate_unparseable_json_exit_2(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert invoke("matrix", "validate", str(path)).exit_code == 2


def test_import_cli_reproduces_committed_example(
    tmp_path: Path, example_csv_path: Path, example_matrix_path: Path
) -> None:
    out = tmp_path / "matrix.json"
    result = invoke(
        "matrix",
        "import",
        "--csv",
        str(example_csv_path),
        "-o",
        str(out),
        "--litert-version",
        "0.0.0-example",
        "--backend",
        "gpu_mldrift",
        "--device",
        "Example Device",
        "--soc",
        "Example SoC",
        "--generated-at",
        "2026-08-10",
    )
    assert result.exit_code == 0
    assert out.read_bytes() == example_matrix_path.read_bytes()


def test_import_cli_bad_rows_exit_1_nothing_written(tmp_path: Path) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("op,status,provenance\n,delegated,example\n", encoding="utf-8")
    out = tmp_path / "matrix.json"
    result = invoke(
        "matrix",
        "import",
        "--csv",
        str(csv_file),
        "-o",
        str(out),
        "--litert-version",
        "0.0.0-example",
        "--backend",
        "gpu_mldrift",
    )
    assert result.exit_code == 1
    assert not out.exists()


def test_diff_cli_no_change_exit_0(example_matrix_path: Path) -> None:
    result = invoke("matrix", "diff", str(example_matrix_path), str(example_matrix_path))
    assert result.exit_code == 0
    assert "no semantic differences" in result.output


def test_diff_cli_changes_exit_1_and_json(tmp_path: Path) -> None:
    old = write_doc(tmp_path / "old.json", make_doc([entry(status="fallback")]))
    new = write_doc(tmp_path / "new.json", make_doc([entry(status="delegated")]))
    result = invoke("matrix", "diff", str(old), str(new), "--json")
    assert result.exit_code == 1
    import json

    payload = json.loads(result.output)
    assert payload["has_changes"] is True
    assert payload["status_transitions"][0]["from"] == "fallback"


def test_diff_cli_backend_mismatch_exit_2(tmp_path: Path) -> None:
    old = write_doc(tmp_path / "old.json", make_doc([]))
    new = write_doc(tmp_path / "new.json", make_doc([], backend="cpu_xnnpack"))
    assert invoke("matrix", "diff", str(old), str(new)).exit_code == 2


def test_diff_cli_invalid_input_exit_2(tmp_path: Path) -> None:
    old = write_doc(tmp_path / "old.json", make_doc([entry(status="nope")]))
    new = write_doc(tmp_path / "new.json", make_doc([]))
    assert invoke("matrix", "diff", str(old), str(new)).exit_code == 2


def test_carry_forward_cli_end_to_end(tmp_path: Path) -> None:
    source = write_doc(tmp_path / "old.json", measured_doc())
    out = tmp_path / "new.json"
    result = invoke(
        "matrix",
        "carry-forward",
        str(source),
        "--to-litert-version",
        "1.3.0",
        "-o",
        str(out),
        "--generated-at",
        "2026-08-10",
    )
    assert result.exit_code == 0
    assert "1 measured entries downgraded" in result.output
    carried = load_json(out)
    assert validate_document(carried) == []
    assert carried["entries"][0]["provenance"] == "inferred"

    # a release watcher sees the staleness as a semantic change
    diff_result = invoke("matrix", "diff", str(source), str(out))
    assert diff_result.exit_code == 1
    assert "provenance measured → inferred" in diff_result.output


def test_carry_forward_cli_refuses_in_place(tmp_path: Path) -> None:
    source = write_doc(tmp_path / "old.json", measured_doc())
    result = invoke(
        "matrix",
        "carry-forward",
        str(source),
        "--to-litert-version",
        "1.3.0",
        "-o",
        str(source),
    )
    assert result.exit_code == 2


def test_carry_forward_cli_same_version_exit_2(tmp_path: Path) -> None:
    source = write_doc(tmp_path / "old.json", measured_doc())
    result = invoke(
        "matrix",
        "carry-forward",
        str(source),
        "--to-litert-version",
        "1.2.0",
        "-o",
        str(tmp_path / "new.json"),
    )
    assert result.exit_code == 2


def test_carry_forward_cli_deterministic(tmp_path: Path) -> None:
    source = write_doc(tmp_path / "old.json", measured_doc())
    outputs = []
    for name in ("a.json", "b.json"):
        out = tmp_path / name
        result = invoke(
            "matrix",
            "carry-forward",
            str(source),
            "--to-litert-version",
            "1.3.0",
            "-o",
            str(out),
            "--generated-at",
            "2026-08-10",
        )
        assert result.exit_code == 0
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]


def test_import_cli_deterministic_across_runs(tmp_path: Path, example_csv_path: Path) -> None:
    outputs = []
    for name in ("a.json", "b.json"):
        out = tmp_path / name
        result = invoke(
            "matrix",
            "import",
            "--csv",
            str(example_csv_path),
            "-o",
            str(out),
            "--litert-version",
            "0.0.0-example",
            "--backend",
            "gpu_mldrift",
            "--generated-at",
            "2026-08-10",
        )
        assert result.exit_code == 0
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]


def test_diff_json_matches_canonical_form(tmp_path: Path) -> None:
    old = write_doc(tmp_path / "old.json", make_doc([]))
    new = write_doc(tmp_path / "new.json", make_doc([entry()]))
    result = invoke("matrix", "diff", str(old), str(new), "--json")
    assert result.exit_code == 1
    import json

    payload = json.loads(result.output)
    assert result.output == canonical_dumps(payload)
