"""The LiteRT #10445 class made matchable (DECISIONS #178):

- the reader keeps small constant payloads (`TensorInfo.const_data`) so PAD
  paddings can be read without the runtime;
- classify exposes operand ranks, the PAD axis class, the weight dtype/source
  of weighted ops and the consumers of output 0 as non-reported operand_meta;
- the measured gpu_metal_mac rows of 2026-10-01 match the #10231 form (ADD of a
  rank-2 runtime tensor and a rank-3 constant) and its rank-3 fix form;
- the mixed-rank-elementwise-const-to-rank3 rule rewrites the fixture to
  RESHAPE -> ADD, bit-exact on the CPU reference, and leaves the already-lifted
  form alone.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from litert_compat.fix.engine import run_fix
from litert_compat.fix.rules import load_rule_file
from litert_compat.lint.classify import GraphIndex, classify_model, operand_meta, pad_axis
from litert_compat.matrix.query import Matrix
from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    OpSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.parser.reader import parse_tflite
from litert_compat.probe.runners import CpuRunner, Tolerance

TRANSFORMS = REPO_ROOT / "data" / "transforms"
RULE = TRANSFORMS / "mixed-rank-elementwise-const-to-rank3.json"
FIXTURE_MODEL = TRANSFORMS / "fixtures" / "model_mixed_rank_add_const_fixture.tflite"
FIXTURE_MATRIX = TRANSFORMS / "fixtures" / "matrix_gpu_mixed_rank_add_fixture.json"
METAL_220 = REPO_ROOT / "data" / "matrix" / "gpu_metal_mac__2.2.0.json"
METAL_216 = REPO_ROOT / "data" / "matrix" / "gpu_metal_mac__2.1.6.json"
RUNNERS_AVAILABLE = CpuRunner().availability() is None
ADD_OPTIONS = BuiltinOptionsSpec(type_code=11)


def _add(x_shape: tuple[int, ...], b_shape: tuple[int, ...], const_b: bool = True) -> bytes:
    n = 1
    for d in b_shape:
        n *= d
    data = struct.pack(f"<{n}f", *[0.5 - 0.1 * (i % 7) for i in range(n)]) if const_b else None
    out_shape = b_shape if len(b_shape) >= len(x_shape) else x_shape
    tensors = (
        TensorSpec("x", "float32", x_shape),
        TensorSpec("b", "float32", b_shape, data=data),
        TensorSpec("y", "float32", out_shape),
    )
    ops = (OpSpec("ADD", (0, 1), (2,), builtin_options=ADD_OPTIONS),)
    inputs = (0,) if const_b else (0, 1)
    return build_tflite(GraphSpec(tensors, ops, inputs=inputs, outputs=(2,)))


def _pad(shape: tuple[int, ...], pads: list[tuple[int, int]] | None, dtype: str = "int32") -> bytes:
    rank = len(shape)
    out = tuple(d + (pads[i][0] + pads[i][1] if pads else 0) for i, d in enumerate(shape))
    fmt = "i" if dtype == "int32" else "q"
    data = (
        struct.pack(f"<{2 * rank}{fmt}", *[v for lo_hi in pads for v in lo_hi]) if pads else None
    )
    tensors = (
        TensorSpec("x", "float32", shape),
        TensorSpec("paddings", dtype, (rank, 2), data=data),
        TensorSpec("y", "float32", out),
    )
    ops = (OpSpec("PAD", (0, 1), (2,), builtin_options=BuiltinOptionsSpec(type_code=22)),)
    inputs = (0,) if pads else (0, 1)
    return build_tflite(GraphSpec(tensors, ops, inputs=inputs, outputs=(2,)))


def _first_meta(model_bytes: bytes, node_index: int = 0) -> dict:
    sg = parse_tflite(model_bytes).subgraphs[0]
    return operand_meta(sg, sg.nodes[node_index], GraphIndex.of(sg))


# --- reader: small constant payloads ------------------------------------------------------------


def test_reader_keeps_small_constant_payloads_only() -> None:
    sg = parse_tflite(_pad((16, 32, 1), [(0, 0), (0, 32), (0, 0)])).subgraphs[0]
    x, paddings, y = sg.tensors
    assert paddings.is_constant and paddings.const_data is not None
    assert struct.unpack("<6i", paddings.const_data) == (0, 0, 0, 32, 0, 0)
    assert x.const_data is None and y.const_data is None
    big = _add((2, 2048), (1, 2, 2048))  # 16 KiB constant: data-bearing but not copied
    b = parse_tflite(big).subgraphs[0].tensors[1]
    assert b.is_constant and b.const_data is None


# --- classify: operand ranks, PAD axis, weights, consumers --------------------------------------


def test_operand_meta_carries_ranks_and_consumers() -> None:
    meta = _first_meta(_add((8, 4), (1, 8, 4)))
    assert meta["operand_a"] == "activation" and meta["operand_a_rank"] == 2
    assert meta["operand_b"] == "constant" and meta["operand_b_rank"] == 3
    assert meta["consumer_ops"] == "none"


def test_pad_axis_classes() -> None:
    assert _first_meta(_pad((16, 32, 1), [(0, 0), (0, 32), (0, 0)]))["axis"] == "non-innermost"
    assert _first_meta(_pad((16, 32, 8), [(0, 4), (0, 0), (0, 0)]))["axis"] == "non-innermost"
    assert _first_meta(_pad((1, 6144, 32), [(0, 0), (0, 0), (4, 0)]))["axis"] == "innermost"
    int64 = _first_meta(_pad((1, 6144, 32), [(0, 0), (0, 0), (0, 4)], "int64"))
    assert int64["axis"] == "innermost"
    assert _first_meta(_pad((16, 32, 1), [(0, 0), (0, 0), (0, 0)]))["axis"] == "none"
    assert _first_meta(_pad((16, 32, 1), None))["axis"] == "unknown"  # runtime paddings
    sg = parse_tflite(_pad((16, 32, 1), [(0, 0), (0, 32), (0, 0)])).subgraphs[0]
    assert pad_axis(sg, sg.nodes[0]) == "non-innermost"


def test_pad_row_of_the_metal_matrix_matches_the_9272_form() -> None:
    """The gpu_metal_mac__2.2.0 PAD row (`rank: 3, axis: non-innermost`) was
    documentation-only before the `axis` key; it now classifies the #9272 graph."""
    matrix = Matrix.load(METAL_220)
    mid = classify_model(parse_tflite(_pad((16, 32, 1), [(0, 0), (0, 32), (0, 0)])), matrix)
    inner = classify_model(parse_tflite(_pad((1, 6144, 32), [(0, 0), (0, 0), (0, 4)])), matrix)
    assert mid[0].verdict.status == "incorrect"
    assert inner[0].verdict.status == "unknown"
    assert "axis" not in mid[0].shape_meta  # reports stay pinned


def _fc(weight_dtype: str, explicit_dequantize: bool, feed_add: bool) -> bytes:
    k, n = 4, 3
    if weight_dtype == "int8":
        w_bytes = bytes(range(k * n))
    else:
        w_bytes = struct.pack(f"<{k * n}f", *([0.5] * (k * n)))
    tensors = [
        TensorSpec("x", "float32", (1, 5, k)),
        TensorSpec("w_q", weight_dtype, (n, k), data=w_bytes),
    ]
    ops: list[OpSpec] = []
    w_index = 1
    if explicit_dequantize:
        tensors.append(TensorSpec("w", "float32", (n, k)))
        ops.append(OpSpec("DEQUANTIZE", (1,), (2,)))
        w_index = 2
    y_index = len(tensors)
    tensors.append(TensorSpec("y", "float32", (1, 5, n)))
    ops.append(OpSpec("FULLY_CONNECTED", (0, w_index), (y_index,),
                      builtin_options=BuiltinOptionsSpec(type_code=8)))
    outputs = (y_index,)
    if feed_add:
        tensors.append(TensorSpec("z", "float32", (1, 5, n)))
        ops.append(OpSpec("ADD", (y_index, y_index), (y_index + 1,), builtin_options=ADD_OPTIONS))
        outputs = (y_index + 1,)
    return build_tflite(GraphSpec(tuple(tensors), tuple(ops), inputs=(0,), outputs=outputs))


def test_weighted_op_keys_and_consumers() -> None:
    drq = _first_meta(_fc("int8", explicit_dequantize=False, feed_add=True))
    assert drq["weight_dtype"] == "int8" and drq["weight_source"] == "constant"
    assert drq["consumer_ops"] == "ADD" and drq["operand_a_rank"] == 3
    deq = _first_meta(_fc("int8", explicit_dequantize=True, feed_add=False), node_index=1)
    assert deq["weight_dtype"] == "int8" and deq["weight_source"] == "DEQUANTIZE"
    assert deq["consumer_ops"] == "none"
    f32 = _first_meta(_fc("float32", explicit_dequantize=False, feed_add=False))
    assert f32["weight_dtype"] == "float32" and f32["weight_source"] == "constant"


# --- the measured rows and the rule --------------------------------------------------------------


@pytest.mark.parametrize("matrix_path", [METAL_220, METAL_216])
def test_metal_rows_match_the_10231_form_and_its_fix(matrix_path: Path) -> None:
    matrix = Matrix.load(matrix_path)
    mixed = classify_model(parse_tflite(_add((85, 16), (1, 85, 16))), matrix)
    lifted = classify_model(parse_tflite(_add((1, 85, 16), (1, 85, 16))), matrix)
    two_inputs = classify_model(parse_tflite(_add((85, 16), (1, 85, 16), const_b=False)), matrix)
    assert mixed[0].verdict.status == "incorrect"
    assert lifted[0].verdict.status == "delegated"
    assert two_inputs[0].verdict.status == "unknown"  # operand_b activation: not the measured form
    hint = mixed[0].verdict.matched_entry["rewrite_hints"][0]
    assert hint["transform_id"] == "mixed-rank-elementwise-const-to-rank3"
    # the same-shape rank-2 constant row (SAM 2.1 site) is a different row: it still matches
    # on 2.1.6; the 2.2.0 row pins `shape: "[4096,256]"`, a key the linter does not compute,
    # so it stays documentation-only there (unchanged by this change).
    same_rank2 = classify_model(parse_tflite(_add((8, 4), (8, 4))), matrix)
    if matrix_path == METAL_216:
        assert same_rank2[0].verdict.status == "incorrect"
        assert same_rank2[0].verdict.matched_entry["constraints"]["rank"] == 2
    else:
        assert same_rank2[0].verdict.status == "unknown"


def test_fixture_is_builder_output_and_rule_loads() -> None:
    import sys

    sys.path.insert(0, str(TRANSFORMS / "fixtures"))
    try:
        import make_fixtures
    finally:
        sys.path.pop(0)
    assert FIXTURE_MODEL.read_bytes() == make_fixtures.fixture_mixed_rank_add_const()
    rule = load_rule_file(RULE)
    assert rule.action == "decompose" and rule.provenance == "measured"


def _fix(model_bytes: bytes, tmp_path: Path, matrix_path: Path = FIXTURE_MATRIX):
    model = tmp_path / "m.tflite"
    model.write_bytes(model_bytes)
    return run_fix(
        model_path=model, matrix=Matrix.load(matrix_path), rules=[load_rule_file(RULE)],
        matrix_path=str(matrix_path), rules_dir=str(TRANSFORMS), mode="dry_run",
        tolerance=Tolerance(),
    ).report


@pytest.mark.skipif(not RUNNERS_AVAILABLE, reason="CPU runner extra not installed")
def test_mixed_rank_rule_fixes_fixture_bit_exact(tmp_path: Path) -> None:
    report = _fix(FIXTURE_MODEL.read_bytes(), tmp_path)
    assert report["outcome"] == "fixed", json.dumps(report, indent=1)[:2000]
    assert report["before"]["summary"]["status_counts"] == {"incorrect": 1}
    assert report["after"]["summary"]["status_counts"] == {"delegated": 2}
    assert report["plan"][0]["detail"] == "ADD → RESHAPE + ADD"
    assert report["verification"]["status"] == "passed"
    assert report["verification"]["max_abs_diff"] == 0.0


@pytest.mark.skipif(not RUNNERS_AVAILABLE, reason="CPU runner extra not installed")
def test_mixed_rank_rule_against_the_real_metal_snapshot(tmp_path: Path) -> None:
    """The measured 2.2.0 rows carry the fix through the improvement gate on
    the real snapshot: the RESHAPE and the rank-3 ADD are both `delegated` rows."""
    report = _fix(_add((85, 16), (1, 85, 16)), tmp_path, matrix_path=METAL_220)
    assert report["outcome"] == "fixed", json.dumps(report, indent=1)[:2000]
    assert report["after"]["summary"]["status_counts"] == {"delegated": 2}
    assert report["verification"]["max_abs_diff"] == 0.0


def test_mixed_rank_rule_leaves_the_lifted_form_alone(tmp_path: Path) -> None:
    report = _fix(_add((1, 85, 16), (1, 85, 16)), tmp_path)
    assert report["outcome"] == "nothing_to_fix"
    assert report["plan"] == []
