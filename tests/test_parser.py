"""Parser: builder round trip, fixture regeneration, error paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO_ROOT
from litert_compat.parser import (
    GraphSpec,
    OpSpec,
    TensorSpec,
    TfliteParseError,
    build_tflite,
    parse_tflite,
)
from litert_compat.parser.fixtures import FIXTURES
from litert_compat.parser.opcodes import builtin_name, tensor_type_name


def tiny_graph(**overrides) -> GraphSpec:
    spec = {
        "tensors": (
            TensorSpec("in", "float32", (1, 4)),
            TensorSpec("out", "float32", (1, 4)),
        ),
        "ops": (OpSpec("TANH", (0,), (1,)),),
        "inputs": (0,),
        "outputs": (1,),
    }
    spec.update(overrides)
    return GraphSpec(**spec)


def test_committed_fixtures_are_builder_output() -> None:
    """The checked-in .tflite fixtures byte-match regeneration (determinism
    plus provenance: fixtures are exactly what fixtures.py builds)."""
    for name, data in FIXTURES.items():
        committed = (REPO_ROOT / "data" / "examples" / name).read_bytes()
        assert committed == data, f"{name} out of date: python -m litert_compat.parser.fixtures"


def test_round_trip_ops_tensors_shapes() -> None:
    model = parse_tflite(build_tflite(tiny_graph()))
    assert len(model.subgraphs) == 1
    sg = model.subgraphs[0]
    assert [n.op for n in sg.nodes] == ["TANH"]
    assert sg.nodes[0].inputs == (0,) and sg.nodes[0].outputs == (1,)
    assert sg.tensors[0].dtype == "float32"
    assert sg.tensors[0].shape == (1, 4)
    assert sg.tensors[0].name == "in"
    assert sg.inputs == (0,) and sg.outputs == (1,)


def test_round_trip_custom_op() -> None:
    graph = tiny_graph(ops=(OpSpec("CUSTOM", (0,), (1,), custom_code="MyOp"),))
    node = parse_tflite(build_tflite(graph)).subgraphs[0].nodes[0]
    assert node.op == "CUSTOM"
    assert node.custom_code == "MyOp"


def test_round_trip_op_version_and_high_opcode() -> None:
    # GELU (150) only exists in the extended builtin_code field (> 127).
    graph = tiny_graph(ops=(OpSpec("GELU", (0,), (1,), version=2),))
    node = parse_tflite(build_tflite(graph)).subgraphs[0].nodes[0]
    assert node.op == "GELU"
    assert node.op_version == 2


def test_shape_signature_marks_dynamic() -> None:
    graph = tiny_graph(
        tensors=(
            TensorSpec("in", "float32", (1, 4), shape_signature=(-1, 4)),
            TensorSpec("out", "float32", (1, 4)),
        )
    )
    sg = parse_tflite(build_tflite(graph)).subgraphs[0]
    assert sg.tensors[0].is_dynamic is True
    assert sg.tensors[0].rank == 2
    assert sg.tensors[1].is_dynamic is False


def test_builder_deterministic() -> None:
    assert build_tflite(tiny_graph()) == build_tflite(tiny_graph())


def test_builder_rejects_bad_specs() -> None:
    with pytest.raises(ValueError, match="unknown builtin"):
        build_tflite(tiny_graph(ops=(OpSpec("NOT_AN_OP", (0,), (1,)),)))
    with pytest.raises(ValueError, match="custom_code"):
        build_tflite(tiny_graph(ops=(OpSpec("CUSTOM", (0,), (1,)),)))
    with pytest.raises(ValueError, match="out of range"):
        build_tflite(tiny_graph(ops=(OpSpec("TANH", (0,), (9,)),)))
    with pytest.raises(ValueError, match="unknown dtype"):
        build_tflite(
            tiny_graph(
                tensors=(
                    TensorSpec("in", "float99", (1,)),
                    TensorSpec("out", "float32", (1,)),
                )
            )
        )


def test_parse_rejects_non_tflite(tmp_path: Path) -> None:
    with pytest.raises(TfliteParseError, match="too small"):
        parse_tflite(b"\x00")
    with pytest.raises(TfliteParseError, match="file identifier"):
        parse_tflite(b"\x00\x00\x00\x00NOPE" + b"\x00" * 64)


def test_parse_rejects_truncated_fixture() -> None:
    data = FIXTURES["model_clean_example.tflite"]
    with pytest.raises(TfliteParseError):
        parse_tflite(data[: len(data) // 3])


def test_unknown_vocabulary_render() -> None:
    assert builtin_name(3) == "CONV_2D"
    assert builtin_name(99999) == "UNKNOWN_BUILTIN_99999"
    assert tensor_type_name(0) == "float32"
    assert tensor_type_name(250) == "unknown_dtype_250"
