"""The rank-0 operand class made matchable (DECISIONS #236):

- a reduction without `keepdim` puts a rank-0 tensor into broadcast arithmetic;
  ML Drift accepts the ADD and computes it wrong (granite-4.0-h-1b, 2026-08-13,
  Mali OpenCL and Mac Metal) or refuses DIV / MUL with "has bad input dims
  size: 0" (VibeVoice-ASR-BitNet encoder, 2026-09-03);
- classify exposes `min_operand_rank` (smallest known input rank) as
  non-reported operand_meta, so the measured rows can name the form as
  `min_operand_rank: 0` whichever side the scalar is on;
- the four measured rows carry that constraint instead of `null`, so a plain
  rank-1 ADD / DIV / MUL on those snapshots is no longer claimed by them.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from litert_compat.lint.classify import GraphIndex, classify_model, operand_meta
from litert_compat.matrix.query import Matrix
from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    OpSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.parser.reader import parse_tflite

MATRIX = REPO_ROOT / "data" / "matrix"
ANDROID_0160 = MATRIX / "gpu_mldrift_android__0.16.0.json"
METAL_0160 = MATRIX / "gpu_metal_mac__0.16.0.json"
METAL_0161 = MATRIX / "gpu_metal_mac__0.16.1.json"
OPTIONS = {"ADD": BuiltinOptionsSpec(type_code=11), "MUL": BuiltinOptionsSpec(type_code=21),
           "DIV": BuiltinOptionsSpec(type_code=42)}


def _binary(op: str, a_shape: tuple[int, ...], b_shape: tuple[int, ...],
            const_b: bool = False) -> bytes:
    """`op(a, b)` with both operands float32; `b` optionally a constant."""
    n = 1
    for d in b_shape:
        n *= d
    data = struct.pack(f"<{n}f", *[1.0 + i for i in range(n)]) if const_b else None
    out_shape = a_shape if len(a_shape) >= len(b_shape) else b_shape
    tensors = (
        TensorSpec("a", "float32", a_shape),
        TensorSpec("b", "float32", b_shape, data=data),
        TensorSpec("y", "float32", out_shape),
    )
    ops = (OpSpec(op, (0, 1), (2,), builtin_options=OPTIONS[op]),)
    inputs = (0,) if const_b else (0, 1)
    return build_tflite(GraphSpec(tensors, ops, inputs=inputs, outputs=(2,)))


def _meta(model_bytes: bytes) -> dict:
    sg = parse_tflite(model_bytes).subgraphs[0]
    return operand_meta(sg, sg.nodes[0], GraphIndex.of(sg))


def _status(model_bytes: bytes, matrix_path: Path) -> str:
    return classify_model(parse_tflite(model_bytes), Matrix.load(matrix_path))[0].verdict.status


def test_min_operand_rank_is_the_smallest_known_input_rank() -> None:
    scalar_left = _meta(_binary("ADD", (), (4,), const_b=True))
    assert scalar_left["operand_a_rank"] == 0 and scalar_left["operand_b_rank"] == 1
    assert scalar_left["min_operand_rank"] == 0
    scalar_right = _meta(_binary("DIV", (1, 8), ()))
    assert scalar_right["min_operand_rank"] == 0
    assert _meta(_binary("ADD", (8, 4), (1, 8, 4), const_b=True))["min_operand_rank"] == 2
    assert _meta(_binary("MUL", (4,), (4,)))["min_operand_rank"] == 1


@pytest.mark.parametrize("matrix_path", [ANDROID_0160, METAL_0160])
def test_rank0_broadcast_add_matches_the_granite_row(matrix_path: Path) -> None:
    """`rank0_scalar + arange`: accepted and wrong on both delegates (catalog D3)."""
    assert _status(_binary("ADD", (), (4,), const_b=True), matrix_path) == "incorrect"
    assert _status(_binary("ADD", (4,), ()), matrix_path) == "incorrect"  # scalar on the right
    # A plain float32 ADD of ranked operands is not the measured form: the row no
    # longer claims it (it was claimed while the row carried `constraints: null`).
    assert _status(_binary("ADD", (4,), (4,)), matrix_path) != "incorrect"


def test_rank0_div_and_mul_match_the_vibevoice_rows() -> None:
    """`x / rms` with a rank-0 statistic: refused, "has bad input dims size: 0" (catalog D16)."""
    assert _status(_binary("DIV", (1, 8), ()), METAL_0161) == "fallback"
    assert _status(_binary("MUL", (), (1, 8)), METAL_0161) == "fallback"
    assert _status(_binary("DIV", (1, 8), (1, 8)), METAL_0161) != "fallback"
    assert _status(_binary("MUL", (1, 8), (1, 1)), METAL_0161) != "fallback"


@pytest.mark.parametrize(
    ("matrix_path", "op"),
    [(ANDROID_0160, "ADD"), (METAL_0160, "ADD"), (METAL_0161, "DIV"), (METAL_0161, "MUL")],
)
def test_the_measured_rows_name_the_rank0_form(matrix_path: Path, op: str) -> None:
    doc = Matrix.load(matrix_path).doc
    rows = [
        e for e in doc["entries"]
        if e["op"] == op and "rank" in (e.get("conditions") or "").lower()
        and "0" in (e.get("conditions") or "")
    ]
    assert rows, f"{matrix_path.name}: no rank-0 {op} row"
    for row in rows:
        assert row["constraints"] == {"min_operand_rank": 0}
        assert row["provenance"] == "measured"
