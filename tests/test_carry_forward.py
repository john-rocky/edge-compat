from __future__ import annotations

import pytest

from helpers import entry, make_doc
from litert_compat.matrix import carry_forward, diff_documents, validate_document
from litert_compat.matrix.canonical import canonical_dumps


def measured_entry(**fields: object) -> dict:
    return entry(
        provenance="measured",
        evidence={"source_model": "model-a", "litert_version": "1.2.0", "date": "2026-08-01"},
        **fields,
    )


def source_doc() -> dict:
    return make_doc(
        [
            measured_entry(),
            entry(op="ADD", provenance="vendor_doc"),
            entry(op="SUB", provenance="example"),
            entry(
                op="MUL",
                provenance="inferred",
                evidence={"derived_from": {"litert_version": "1.1.0", "date": "2026-07-01"}},
            ),
        ],
        litert_version="1.2.0",
        generated_at="2026-08-01",
    )


def test_measured_downgrades_to_inferred_with_derived_from() -> None:
    doc = source_doc()
    out = carry_forward(doc, to_litert_version="1.3.0", generated_at="2026-08-10")
    carried = next(e for e in out["entries"] if e["op"] == "CONV_2D")
    assert carried["provenance"] == "inferred"
    assert carried["evidence"]["derived_from"] == {
        "litert_version": "1.2.0",
        "date": "2026-08-01",
    }
    assert carried["evidence"]["source_model"] == "model-a"  # original evidence preserved
    assert out["litert_version"] == "1.3.0"
    assert out["generated_at"] == "2026-08-10"


def test_input_document_is_not_mutated() -> None:
    doc = source_doc()
    carry_forward(doc, to_litert_version="1.3.0", generated_at="2026-08-10")
    assert next(e for e in doc["entries"] if e["op"] == "CONV_2D")["provenance"] == "measured"


def test_non_measured_provenances_untouched() -> None:
    out = carry_forward(source_doc(), to_litert_version="1.3.0", generated_at="2026-08-10")
    by_op = {e["op"]: e for e in out["entries"]}
    assert by_op["ADD"]["provenance"] == "vendor_doc"
    assert by_op["SUB"]["provenance"] == "example"
    assert "evidence" not in by_op["ADD"]


def test_already_inferred_keeps_original_derived_from() -> None:
    out = carry_forward(source_doc(), to_litert_version="1.3.0", generated_at="2026-08-10")
    inferred = next(e for e in out["entries"] if e["op"] == "MUL")
    assert inferred["evidence"]["derived_from"] == {
        "litert_version": "1.1.0",
        "date": "2026-07-01",
    }


def test_output_validates_and_is_deterministic() -> None:
    first = carry_forward(source_doc(), to_litert_version="1.3.0", generated_at="2026-08-10")
    second = carry_forward(source_doc(), to_litert_version="1.3.0", generated_at="2026-08-10")
    assert validate_document(first) == []
    assert canonical_dumps(first) == canonical_dumps(second)


def test_same_version_target_rejected() -> None:
    with pytest.raises(ValueError, match="equals the source"):
        carry_forward(source_doc(), to_litert_version="1.2.0", generated_at="2026-08-10")


def test_carry_forward_then_diff_reports_provenance_transitions() -> None:
    doc = source_doc()
    out = carry_forward(doc, to_litert_version="1.3.0", generated_at="2026-08-10")
    result = diff_documents(doc, out)
    assert result["has_changes"] is True
    assert result["status_transitions"] == []
    assert [t["op"] for t in result["provenance_transitions"]] == ["CONV_2D"]
    assert result["provenance_transitions"][0]["from"] == "measured"
    assert result["provenance_transitions"][0]["to"] == "inferred"
