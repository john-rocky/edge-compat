from __future__ import annotations

from pathlib import Path

from helpers import entry, make_doc
from litert_compat.matrix import provenance_counts, validate_document
from litert_compat.matrix.canonical import load_json

ALL_STATUSES = {"delegated", "partial", "fallback", "incorrect", "crash", "unknown"}


def test_example_matrix_passes(example_matrix_path: Path) -> None:
    assert validate_document(load_json(example_matrix_path)) == []


def test_example_matrix_covers_every_status_all_example(example_matrix_path: Path) -> None:
    doc = load_json(example_matrix_path)
    assert {e["status"] for e in doc["entries"]} == ALL_STATUSES
    assert provenance_counts(doc) == {"example": len(doc["entries"])}


def test_measured_without_evidence_is_a_finding() -> None:
    findings = validate_document(make_doc([entry(provenance="measured")]))
    assert any(f.code == "missing_evidence" for f in findings)


def test_measured_with_partial_evidence_is_a_finding() -> None:
    doc = make_doc(
        [entry(provenance="measured", evidence={"source_model": "m"})]
    )
    findings = validate_document(doc)
    assert any(
        f.code == "missing_evidence" and "litert_version" in f.message and "date" in f.message
        for f in findings
    )


def test_measured_with_full_evidence_passes() -> None:
    doc = make_doc(
        [
            entry(
                provenance="measured",
                evidence={
                    "source_model": "m",
                    "litert_version": "1.2.0",
                    "date": "2026-08-01",
                },
            )
        ]
    )
    assert validate_document(doc) == []


def test_contradictory_duplicates_are_a_finding() -> None:
    doc = make_doc([entry(status="delegated"), entry(status="fallback")])
    assert any(f.code == "contradictory_entries" for f in validate_document(doc))


def test_same_status_duplicates_pass() -> None:
    doc = make_doc([entry(conditions="a"), entry(conditions="b")])
    assert validate_document(doc) == []


def test_unknown_status_enum_is_schema_finding() -> None:
    doc = make_doc([entry(status="works_great")])
    assert any(f.code == "schema" for f in validate_document(doc))


def test_unknown_provenance_enum_is_schema_finding() -> None:
    doc = make_doc([entry(provenance="trust_me")])
    assert any(f.code == "schema" for f in validate_document(doc))


def test_missing_top_level_field_is_schema_finding() -> None:
    doc = make_doc([entry()])
    del doc["backend"]
    assert any(f.code == "schema" for f in validate_document(doc))


def test_unknown_dtype_is_schema_finding() -> None:
    doc = make_doc([entry(dtypes=["float999"])])
    assert any(f.code == "schema" for f in validate_document(doc))


def test_provenance_counts_mixed() -> None:
    doc = make_doc(
        [entry(), entry(op="ADD", provenance="vendor_doc"), entry(op="SUB", provenance="inferred")]
    )
    assert provenance_counts(doc) == {"example": 1, "vendor_doc": 1, "inferred": 1}
