from __future__ import annotations

from pathlib import Path

import pytest

from litert_compat.matrix import CsvImportError, import_csv, validate_document
from litert_compat.matrix.canonical import canonical_dumps

EXAMPLE_ARGS = {
    "litert_version": "0.0.0-example",
    "backend": "gpu_mldrift",
    "generated_at": "2026-08-10",
    "target": {"device": "Example Device", "soc": "Example SoC"},
}


def test_import_matches_committed_example(
    example_csv_path: Path, example_matrix_path: Path
) -> None:
    result = import_csv(example_csv_path, **EXAMPLE_ARGS)
    assert canonical_dumps(result.doc) == example_matrix_path.read_text(encoding="utf-8")
    assert result.warnings == []


def test_import_is_byte_deterministic(example_csv_path: Path) -> None:
    first = import_csv(example_csv_path, **EXAMPLE_ARGS)
    second = import_csv(example_csv_path, **EXAMPLE_ARGS)
    assert canonical_dumps(first.doc) == canonical_dumps(second.doc)


def test_import_result_validates(example_csv_path: Path) -> None:
    result = import_csv(example_csv_path, **EXAMPLE_ARGS)
    assert validate_document(result.doc) == []


def test_forgiving_column_order_case_and_whitespace(tmp_path: Path) -> None:
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "Status, OP ,provenance\n delegated , CONV_2D , example \n", encoding="utf-8"
    )
    result = import_csv(
        csv_file, litert_version="0.0.0-example", backend="gpu_mldrift", generated_at="2026-08-10"
    )
    assert result.doc["entries"] == [
        {"op": "CONV_2D", "status": "delegated", "provenance": "example"}
    ]


def test_unknown_columns_warned_and_ignored(tmp_path: Path) -> None:
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "op,status,provenance,notes\nCONV_2D,delegated,example,ignore me\n", encoding="utf-8"
    )
    result = import_csv(
        csv_file, litert_version="0.0.0-example", backend="gpu_mldrift", generated_at="2026-08-10"
    )
    assert any("notes" in w for w in result.warnings)
    assert result.doc["entries"][0] == {
        "op": "CONV_2D",
        "status": "delegated",
        "provenance": "example",
    }


def test_constraint_values_parsed_as_scalars(tmp_path: Path) -> None:
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "op,constraints,status,provenance\n"
        "CONV_2D,dynamic_shape=false;max_rank=4;mode=strict,delegated,example\n",
        encoding="utf-8",
    )
    result = import_csv(
        csv_file, litert_version="0.0.0-example", backend="gpu_mldrift", generated_at="2026-08-10"
    )
    assert result.doc["entries"][0]["constraints"] == {
        "dynamic_shape": False,
        "max_rank": 4,
        "mode": "strict",
    }


def test_row_errors_collected_with_line_numbers(tmp_path: Path) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(
        "op,status,provenance\n"
        ",delegated,example\n"
        "CONV_2D,works_great,example\n"
        "ADD,delegated,trust_me\n",
        encoding="utf-8",
    )
    with pytest.raises(CsvImportError) as exc_info:
        import_csv(
            csv_file,
            litert_version="0.0.0-example",
            backend="gpu_mldrift",
            generated_at="2026-08-10",
        )
    errors = exc_info.value.errors
    assert len(errors) == 3
    assert any("line 2" in e for e in errors)
    assert any("line 3" in e for e in errors)
    assert any("line 4" in e for e in errors)


def test_rewrite_hint_requires_symptom_and_rewrite(tmp_path: Path) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(
        "op,status,provenance,rewrite\nCONV_2D,delegated,example,do something\n",
        encoding="utf-8",
    )
    with pytest.raises(CsvImportError):
        import_csv(
            csv_file,
            litert_version="0.0.0-example",
            backend="gpu_mldrift",
            generated_at="2026-08-10",
        )


def test_empty_rows_skipped(tmp_path: Path) -> None:
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "op,status,provenance\n\n , , \nCONV_2D,delegated,example\n", encoding="utf-8"
    )
    result = import_csv(
        csv_file, litert_version="0.0.0-example", backend="gpu_mldrift", generated_at="2026-08-10"
    )
    assert len(result.doc["entries"]) == 1
