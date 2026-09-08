"""Classification: signature extraction from outputs, claimed-set membership."""

from __future__ import annotations

from pathlib import Path

from litert_compat.lint.classify import classify_model, node_signature
from litert_compat.matrix.query import Matrix
from litert_compat.parser import GraphSpec, OpSpec, TensorSpec, build_tflite, parse_tflite
from litert_compat.parser.fixtures import fixture_mixed, fixture_split


def load_matrix(example_matrix_path: Path) -> Matrix:
    return Matrix.load(example_matrix_path)


def test_signature_uses_output_tensors_only() -> None:
    """TRANSPOSE_CONV's first input is an int32 shape tensor; the signature
    must carry the compute dtype from the outputs, not int32."""
    graph = GraphSpec(
        tensors=(
            TensorSpec("output_shape", "int32", (4,)),
            TensorSpec("filter", "float32", (2, 2, 2, 4)),
            TensorSpec("in", "float16", (1, 4, 4, 4)),
            TensorSpec("out", "float32", (1, 8, 8, 2)),
        ),
        ops=(OpSpec("TRANSPOSE_CONV", (0, 1, 2), (3,)),),
        inputs=(2,),
        outputs=(3,),
    )
    sg = parse_tflite(build_tflite(graph)).subgraphs[0]
    dtypes, shape_meta = node_signature(sg, sg.nodes[0])
    assert dtypes == ["float32"]
    assert shape_meta == {"dynamic_shape": False, "rank": 4}


def test_signature_dynamic_output() -> None:
    graph = GraphSpec(
        tensors=(
            TensorSpec("in", "int8", (1, 64), shape_signature=(-1, 64)),
            TensorSpec("out", "int8", (1, 32), shape_signature=(-1, 32)),
        ),
        ops=(OpSpec("FULLY_CONNECTED", (0,), (1,)),),
        inputs=(0,),
        outputs=(1,),
    )
    sg = parse_tflite(build_tflite(graph)).subgraphs[0]
    dtypes, shape_meta = node_signature(sg, sg.nodes[0])
    assert dtypes == ["int8"]
    assert shape_meta == {"dynamic_shape": True, "rank": 2}


def test_classify_split_fixture(example_matrix_path: Path) -> None:
    classified = classify_model(parse_tflite(fixture_split()), load_matrix(example_matrix_path))
    statuses = [(c.node.op, c.verdict.status, c.claimed) for c in classified]
    assert statuses == [
        ("CONV_2D", "delegated", True),
        ("GATHER_ND", "fallback", False),
        ("RESHAPE", "unknown", False),
        ("FULLY_CONNECTED", "delegated", True),
        ("SOFTMAX", "unknown", False),
    ]
    # every verdict carries the snapshot identity (spec §D)
    assert all(c.verdict.backend == "gpu_mldrift" for c in classified)
    assert all(c.verdict.litert_version == "0.0.0-example" for c in classified)


def test_classify_mixed_fixture_incorrect_is_claimed(example_matrix_path: Path) -> None:
    classified = classify_model(parse_tflite(fixture_mixed()), load_matrix(example_matrix_path))
    by_op = {c.node.op: c for c in classified}
    # incorrect: the delegate runs it (claimed) — the worst failure mode
    assert by_op["SOFTMAX"].verdict.status == "incorrect"
    assert by_op["SOFTMAX"].claimed is True
    # dynamic int8 FULLY_CONNECTED hits the constraint-specific fallback entry
    assert by_op["FULLY_CONNECTED"].verdict.status == "fallback"
    assert by_op["FULLY_CONNECTED"].verdict.matched_entry is not None
    assert by_op["FULLY_CONNECTED"].verdict.matched_entry["rewrite_hints"]
    # custom op: unknown, never claimed, no matrix entry
    custom = by_op["CUSTOM"]
    assert custom.is_custom
    assert custom.verdict.status == "unknown"
    assert custom.verdict.matched_entry is None
    assert "ExampleCustomOp" in custom.verdict.reason
