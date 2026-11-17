from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from helpers import entry, make_doc
from litert_compat.matrix import (
    Matrix,
    MatrixLookupTieError,
    MatrixValidationError,
)


def build(entries: list[dict[str, Any]]) -> Matrix:
    return Matrix(make_doc(entries))


def test_most_specific_match_wins() -> None:
    m = build(
        [
            entry(dtypes=["int8"], constraints={"dynamic_shape": False}, status="delegated"),
            entry(dtypes=["int8"], status="fallback"),
            entry(status="crash"),
        ]
    )
    verdict = m.lookup("CONV_2D", dtypes=["int8"], shape_meta={"dynamic_shape": False})
    assert verdict.status == "delegated"
    assert "constraints" in verdict.reason


def test_dtype_only_beats_op_only() -> None:
    m = build([entry(dtypes=["int8"], status="fallback"), entry(status="delegated")])
    assert m.lookup("CONV_2D", dtypes=["int8"]).status == "fallback"


def test_constraints_only_beats_op_only() -> None:
    m = build(
        [entry(constraints={"dynamic_shape": True}, status="fallback"), entry(status="delegated")]
    )
    assert m.lookup("CONV_2D", shape_meta={"dynamic_shape": True}).status == "fallback"


def test_falls_through_to_op_only_entry() -> None:
    m = build([entry(dtypes=["int8"], status="fallback"), entry(status="delegated")])
    verdict = m.lookup("CONV_2D", dtypes=["float32"])
    assert verdict.status == "delegated"
    assert verdict.reason == "matched op-level entry"


def test_unknown_when_op_absent() -> None:
    m = build([entry()])
    verdict = m.lookup("SOFTMAX")
    assert verdict.status == "unknown"
    assert verdict.matched_entry is None


def test_unknown_when_nothing_matches_never_a_guess() -> None:
    m = build([entry(dtypes=["int8"], status="delegated")])
    verdict = m.lookup("CONV_2D", dtypes=["float32"])
    assert verdict.status == "unknown"
    assert verdict.matched_entry is None


def test_dtype_restricted_entry_does_not_match_dtypeless_query() -> None:
    m = build([entry(dtypes=["int8"], status="delegated")])
    assert m.lookup("CONV_2D").status == "unknown"


def test_max_rank_constraint() -> None:
    m = build([entry(constraints={"max_rank": 4}, status="delegated")])
    assert m.lookup("CONV_2D", shape_meta={"rank": 4}).status == "delegated"
    assert m.lookup("CONV_2D", shape_meta={"rank": 5}).status == "unknown"


def test_min_rank_constraint() -> None:
    m = build([entry(constraints={"min_rank": 2}, status="delegated")])
    assert m.lookup("CONV_2D", shape_meta={"rank": 2}).status == "delegated"
    assert m.lookup("CONV_2D", shape_meta={"rank": 1}).status == "unknown"


def test_missing_shape_meta_never_satisfies_a_constraint() -> None:
    m = build([entry(constraints={"max_rank": 4}, status="delegated")])
    assert m.lookup("CONV_2D").status == "unknown"
    assert m.lookup("CONV_2D", shape_meta={}).status == "unknown"


def test_tie_with_contradictory_statuses_raises() -> None:
    m = build(
        [
            entry(dtypes=["int8"], status="delegated"),
            entry(dtypes=["float32", "int8"], status="fallback"),
        ]
    )
    with pytest.raises(MatrixLookupTieError):
        m.lookup("CONV_2D", dtypes=["int8"])


_MEASURED_EVIDENCE = {"source_model": "m.tflite", "litert_version": "2.3.0", "date": "2026-11-03"}
_CARRIED = {
    "source_model": "probe",
    "litert_version": "2.1.6",
    "date": "2026-08-12",
    "derived_from": {"litert_version": "2.1.6", "date": "2026-08-12"},
}


def test_tie_measured_entry_wins_over_carried_forward_inferred_entry() -> None:
    measured = entry(
        op="BATCH_MATMUL",
        status="crash",
        provenance="measured",
        dtypes=["float32"],
        constraints={"second_input": "constant"},
        evidence=_MEASURED_EVIDENCE,
    )
    m = build(
        [
            entry(
                op="BATCH_MATMUL",
                status="delegated",
                provenance="inferred",
                dtypes=["float32"],
                constraints={"rank": 3},
                evidence=_CARRIED,
            ),
            measured,
        ]
    )
    verdict = m.lookup(
        "BATCH_MATMUL", dtypes=["float32"], shape_meta={"rank": 3, "second_input": "constant"}
    )
    assert verdict.status == "crash"
    assert verdict.matched_entry == measured
    assert verdict.reason == (
        "matched entry specific to op, dtypes, and constraints; tie resolved: measured over "
        'inferred (delegated: BATCH_MATMUL [float32] {"rank":3}, derived from 2.1.6 of 2026-08-12)'
    )


def test_tie_measured_entry_overrules_every_contradicting_inferred_entry() -> None:
    measured = entry(
        status="fallback", provenance="measured", dtypes=["int8"], evidence=_MEASURED_EVIDENCE
    )
    m = build(
        [
            entry(status="delegated", provenance="inferred", dtypes=["float32", "int8"]),
            entry(status="crash", provenance="inferred", dtypes=["int8", "int16"]),
            entry(status="fallback", provenance="inferred", dtypes=["int8", "uint8"]),
            measured,
        ]
    )
    verdict = m.lookup("CONV_2D", dtypes=["int8"])
    assert verdict.status == "fallback"
    assert verdict.matched_entry == measured
    assert verdict.reason.startswith(
        "matched entry specific to op and dtypes; tie resolved: measured over inferred ("
    )
    # Both contradicting entries are named; the agreeing inferred entry is not.
    assert "delegated: CONV_2D [float32,int8] {}" in verdict.reason
    assert "crash: CONV_2D [int16,int8] {}" in verdict.reason
    assert "uint8" not in verdict.reason


def test_tie_between_two_measured_entries_still_raises() -> None:
    m = build(
        [
            entry(
                status="incorrect",
                provenance="measured",
                dtypes=["float32"],
                constraints={"buffer_type": "packed"},
                evidence=_MEASURED_EVIDENCE,
            ),
            entry(
                status="crash",
                provenance="measured",
                dtypes=["float32"],
                constraints={"max_bytes": 1024},
                evidence=_MEASURED_EVIDENCE,
            ),
        ]
    )
    with pytest.raises(MatrixLookupTieError):
        m.lookup(
            "CONV_2D", dtypes=["float32"], shape_meta={"buffer_type": "packed", "max_bytes": 1024}
        )


def test_tie_with_two_measured_entries_against_inferred_still_raises() -> None:
    m = build(
        [
            entry(
                status="crash", provenance="measured", dtypes=["int8"], evidence=_MEASURED_EVIDENCE
            ),
            entry(
                status="crash",
                provenance="measured",
                dtypes=["float32", "int8"],
                evidence=_MEASURED_EVIDENCE,
            ),
            entry(status="delegated", provenance="inferred", dtypes=["int8", "uint8"]),
        ]
    )
    with pytest.raises(MatrixLookupTieError):
        m.lookup("CONV_2D", dtypes=["int8"])


def test_tie_between_inferred_entries_still_raises() -> None:
    m = build(
        [
            entry(status="delegated", provenance="inferred", dtypes=["int8"], evidence=_CARRIED),
            entry(status="fallback", provenance="inferred", dtypes=["float32", "int8"]),
        ]
    )
    with pytest.raises(MatrixLookupTieError):
        m.lookup("CONV_2D", dtypes=["int8"])


def test_tie_measured_against_vendor_doc_still_raises() -> None:
    m = build(
        [
            entry(
                status="crash", provenance="measured", dtypes=["int8"], evidence=_MEASURED_EVIDENCE
            ),
            entry(status="delegated", provenance="vendor_doc", dtypes=["float32", "int8"]),
            entry(status="delegated", provenance="inferred", dtypes=["int8", "uint8"]),
        ]
    )
    with pytest.raises(MatrixLookupTieError):
        m.lookup("CONV_2D", dtypes=["int8"])


def test_tie_with_same_status_is_deterministic() -> None:
    m = build(
        [
            entry(dtypes=["int8"], status="fallback", conditions="a"),
            entry(dtypes=["float32", "int8"], status="fallback"),
        ]
    )
    assert m.lookup("CONV_2D", dtypes=["int8"]).status == "fallback"


def test_verdict_always_carries_backend_and_version(example_matrix_path: Path) -> None:
    m = Matrix.load(example_matrix_path)
    for verdict in (
        m.lookup("CONV_2D", dtypes=["float32"]),
        m.lookup("NOT_A_REAL_OP"),
    ):
        assert verdict.backend == "gpu_mldrift"
        assert verdict.litert_version == "0.0.0-example"


def test_lookup_against_example_file(example_matrix_path: Path) -> None:
    m = Matrix.load(example_matrix_path)
    static = m.lookup(
        "FULLY_CONNECTED", dtypes=["int8"], shape_meta={"dynamic_shape": False, "rank": 2}
    )
    dynamic = m.lookup(
        "FULLY_CONNECTED", dtypes=["int8"], shape_meta={"dynamic_shape": True, "rank": 2}
    )
    assert static.status == "delegated"
    assert dynamic.status == "fallback"
    assert dynamic.matched_entry is not None
    assert dynamic.matched_entry["rewrite_hints"][0]["rewrite"] == (
        "export with a fixed batch dimension"
    )


def test_constructing_matrix_from_invalid_doc_raises() -> None:
    with pytest.raises(MatrixValidationError):
        Matrix(make_doc([entry(status="nope")]))
