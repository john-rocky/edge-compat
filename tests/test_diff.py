from __future__ import annotations

from pathlib import Path

import pytest

from helpers import entry, make_doc
from litert_compat.matrix import BackendMismatchError, diff_documents, render_text
from litert_compat.matrix.canonical import canonical_dumps, load_json


def test_identical_docs_have_no_changes(example_matrix_path: Path) -> None:
    doc = load_json(example_matrix_path)
    result = diff_documents(doc, doc)
    assert result["has_changes"] is False
    for key in (
        "ops_added",
        "ops_removed",
        "entries_added",
        "entries_removed",
        "status_transitions",
        "provenance_transitions",
    ):
        assert result[key] == []
    assert "no semantic differences" in render_text(result)


def test_all_transition_kinds_detected() -> None:
    old = make_doc(
        [
            entry(op="ADD", status="fallback"),
            entry(op="SUB", status="delegated"),
            entry(op="MUL", status="delegated"),
        ],
        litert_version="1.2.0",
    )
    new = make_doc(
        [
            entry(op="ADD", status="delegated"),
            entry(op="SUB", status="delegated", provenance="vendor_doc"),
            entry(op="DIV", status="crash"),
        ],
        litert_version="1.3.0",
    )
    result = diff_documents(old, new)
    assert result["has_changes"] is True
    assert result["ops_added"] == ["DIV"]
    assert result["ops_removed"] == ["MUL"]
    assert result["status_transitions"] == [
        {"op": "ADD", "dtypes": [], "constraints": {}, "from": "fallback", "to": "delegated"}
    ]
    assert result["provenance_transitions"] == [
        {"op": "SUB", "dtypes": [], "constraints": {}, "from": "example", "to": "vendor_doc"}
    ]
    text = render_text(result)
    assert "status fallback → delegated" in text
    assert "provenance example → vendor_doc" in text


def test_dtype_change_is_add_plus_remove_not_transition() -> None:
    old = make_doc([entry(dtypes=["int8"])])
    new = make_doc([entry(dtypes=["float32"])])
    result = diff_documents(old, new)
    assert len(result["entries_added"]) == 1
    assert len(result["entries_removed"]) == 1
    assert result["status_transitions"] == []
    assert result["ops_added"] == []  # op still present on both sides
    assert result["ops_removed"] == []


def test_free_text_edits_are_not_semantic() -> None:
    old = make_doc([entry(conditions="old wording")])
    new = make_doc([entry(conditions="new wording")])
    assert diff_documents(old, new)["has_changes"] is False


def test_backend_mismatch_raises() -> None:
    with pytest.raises(BackendMismatchError):
        diff_documents(make_doc([]), make_doc([], backend="cpu_xnnpack"))


def test_diff_output_is_deterministic() -> None:
    old = make_doc([])
    new = make_doc(
        [entry(op="ZETA"), entry(op="ALPHA"), entry(op="MID")],
    )
    first = canonical_dumps(diff_documents(old, new))
    second = canonical_dumps(diff_documents(old, new))
    assert first == second
    added_ops = [item["op"] for item in diff_documents(old, new)["entries_added"]]
    assert added_ops == sorted(added_ops)
