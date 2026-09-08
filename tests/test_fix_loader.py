"""Fix loader: byte-identical round-trips on builder output, precise refusals."""

from __future__ import annotations

import flatbuffers
import pytest

from litert_compat.fix.loader import UnsupportedModelError, load_graph_spec
from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    MetadataSpec,
    OpSpec,
    OptionsField,
    QuantizationSpec,
    SignatureDefSpec,
    TensorMapSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.parser.fixtures import FIXTURES
from litert_compat.parser.reader import TfliteParseError
from litert_compat.probe.gen import build_probe_graph
from litert_compat.probe.signatures import make_signature

ROUND_TRIP_FIXTURES = sorted(
    name
    for name in FIXTURES
    if name not in ("model_broken_example.tflite", "model_mixed_example.tflite")
)


@pytest.mark.parametrize("name", ROUND_TRIP_FIXTURES)
def test_round_trip_is_byte_identical_on_builder_fixtures(name: str) -> None:
    data = FIXTURES[name]
    assert build_tflite(load_graph_spec(data)) == data


@pytest.mark.parametrize(
    ("op", "meta"),
    [
        ("SOFTMAX", {"dynamic_shape": False, "rank": 2}),  # float options field
        ("ADD", {"dynamic_shape": False, "rank": 4}),  # empty options table
        ("RESHAPE", {"dynamic_shape": False, "rank": 3}),  # constant data buffer
        ("NEG", {"dynamic_shape": True, "rank": 2}),  # shape_signature
        # Weighted/structured templates: options layouts registered for load
        ("CONV_2D", {"dynamic_shape": False, "rank": 4}),
        ("DEPTHWISE_CONV_2D", {"dynamic_shape": False, "rank": 4}),
        ("FULLY_CONNECTED", {"dynamic_shape": False, "rank": 2}),
        ("MAX_POOL_2D", {"dynamic_shape": False, "rank": 4}),
        ("TRANSPOSE_CONV", {"dynamic_shape": False, "rank": 4}),
        ("MEAN", {"dynamic_shape": False, "rank": 4}),
        # Union type codes corrected against real converter output (2026-08-12)
        ("LEAKY_RELU", {"dynamic_shape": False, "rank": 2}),
        ("FLOOR_MOD", {"dynamic_shape": False, "rank": 2}),
        ("SQUARED_DIFFERENCE", {"dynamic_shape": False, "rank": 2}),
        ("RESIZE_NEAREST_NEIGHBOR", {"dynamic_shape": False, "rank": 4}),
        ("SELECT_V2", {"dynamic_shape": False, "rank": 2}),
        ("BATCH_MATMUL", {"dynamic_shape": False, "rank": 3}),
    ],
)
def test_round_trip_is_byte_identical_on_probe_fixtures(op: str, meta: dict) -> None:
    sig = make_signature("gpu_mldrift", op, ["float32"], meta)
    data = build_tflite(build_probe_graph(sig))
    assert build_tflite(load_graph_spec(data)) == data


@pytest.mark.parametrize("op", ["CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "RESHAPE"])
def test_round_trip_is_byte_identical_on_int8_probe_fixtures(op: str) -> None:
    rank = 2 if op == "RESHAPE" else 4
    sig = make_signature("cpu_xnnpack", op, ["int8"], {"dynamic_shape": False, "rank": rank})
    data = build_tflite(build_probe_graph(sig))
    assert build_tflite(load_graph_spec(data)) == data


def test_custom_op_model_is_refused() -> None:
    with pytest.raises(UnsupportedModelError, match="custom op"):
        load_graph_spec(FIXTURES["model_mixed_example.tflite"])


def test_not_a_flatbuffer_raises_parse_error() -> None:
    with pytest.raises(TfliteParseError):
        load_graph_spec(FIXTURES["model_broken_example.tflite"])


def _tiny(options: BuiltinOptionsSpec | None) -> bytes:
    return build_tflite(
        GraphSpec(
            tensors=(
                TensorSpec("in", "float32", (1, 4)),
                TensorSpec("out", "float32", (1, 4)),
            ),
            ops=(OpSpec("TANH", (0,), (1,), builtin_options=options),),
            inputs=(0,),
            outputs=(1,),
        )
    )


def test_unregistered_options_type_is_refused() -> None:
    # LSTMOptions (union type 14) is outside the v1 options registry.
    data = _tiny(BuiltinOptionsSpec(type_code=14, fields=(OptionsField(0, "int8", 1),)))
    with pytest.raises(UnsupportedModelError, match=r"outside the .* options registry"):
        load_graph_spec(data)


def test_populated_field_outside_registered_layout_is_refused() -> None:
    # SoftmaxOptions is registered, but only slot 0 (beta) — slot 3 is not.
    data = _tiny(BuiltinOptionsSpec(type_code=9, fields=(OptionsField(3, "int32", 7),)))
    with pytest.raises(UnsupportedModelError, match="outside the registered layout"):
        load_graph_spec(data)


# --- real-converter fidelity: quantization, metadata, signatures, -1 inputs --


def _fidelity_graph() -> GraphSpec:
    # Float values chosen to be exactly representable in float32 so the
    # loaded spec compares equal to the source spec.
    return GraphSpec(
        tensors=(
            TensorSpec(
                "in", "int8", (1, 4),
                quantization=QuantizationSpec(scale=(0.5,), zero_point=(-3,)),
            ),
            TensorSpec(
                "w", "int8", (4, 4), data=bytes(range(16)), has_rank=True,
                quantization=QuantizationSpec(
                    scale=(0.25, 0.5, 1.5, 2.0),
                    zero_point=(0, 1, -1, 2),
                    quantized_dimension=0,
                ),
            ),
            TensorSpec(
                "out", "int8", (1, 4),
                quantization=QuantizationSpec(
                    scale=(1.0,), zero_point=(2,), min=(-1.0,), max=(1.0,)
                ),
            ),
        ),
        # -1 marks the deliberately absent optional bias input.
        ops=(OpSpec("FULLY_CONNECTED", (0, 1, -1), (2,),
                    builtin_options=BuiltinOptionsSpec(type_code=8)),),
        inputs=(0,),
        outputs=(2,),
        metadata=(
            MetadataSpec("min_runtime_version", b"1.14.0\x00\x00"),
            MetadataSpec("CONVERSION_METADATA", b"\x01\x02\x03"),
        ),
        signature_defs=(
            SignatureDefSpec(
                "serving_default",
                inputs=(TensorMapSpec("x", 0),),
                outputs=(TensorMapSpec("y", 2),),
            ),
        ),
    )


def test_fidelity_fields_round_trip_byte_identical_and_spec_equal() -> None:
    graph = _fidelity_graph()
    data = build_tflite(graph)
    loaded = load_graph_spec(data)
    assert loaded == graph
    assert build_tflite(loaded) == data


def _foreign_model(
    *, quant_details: bool = False, metadata_buffer: bool = False,
    deprecated_tag: bool = False,
) -> bytes:
    """Hand-rolled flatbuffer with a feature our builder refuses to emit."""
    b = flatbuffers.Builder(256)
    b.StartObject(1)
    empty_buffer = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(empty_buffer)
    buffers = b.EndVector()

    name = b.CreateString("t")
    b.StartVector(4, 1, 4)
    b.PrependInt32(1)
    shape = b.EndVector()
    quant = None
    if quant_details:
        b.StartObject(7)
        b.PrependUint8Slot(4, 1, 0)  # details_type: custom quantization
        quant = b.EndObject()
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, shape, 0)
    b.PrependUOffsetTRelativeSlot(3, name, 0)
    if quant is not None:
        b.PrependUOffsetTRelativeSlot(4, quant, 0)
    tensor = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(tensor)
    tensors = b.EndVector()

    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, tensors, 0)
    subgraph = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(subgraph)
    subgraphs = b.EndVector()

    metadata_buffer_vector = None
    if metadata_buffer:
        b.StartVector(4, 1, 4)
        b.PrependInt32(0)
        metadata_buffer_vector = b.EndVector()
    signature_vector = None
    if deprecated_tag:
        tag = b.CreateString("serve")
        b.StartObject(5)
        b.PrependUOffsetTRelativeSlot(3, tag, 0)
        signature = b.EndObject()
        b.StartVector(4, 1, 4)
        b.PrependUOffsetTRelative(signature)
        signature_vector = b.EndVector()

    b.StartObject(8)
    b.PrependUint32Slot(0, 3, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraphs, 0)
    b.PrependUOffsetTRelativeSlot(4, buffers, 0)
    if metadata_buffer_vector is not None:
        b.PrependUOffsetTRelativeSlot(5, metadata_buffer_vector, 0)
    if signature_vector is not None:
        b.PrependUOffsetTRelativeSlot(7, signature_vector, 0)
    model = b.EndObject()
    b.Finish(model, file_identifier=b"TFL3")
    return bytes(b.Output())


@pytest.mark.parametrize(
    ("knob", "match"),
    [
        ({"quant_details": True}, "details_type"),
        ({"metadata_buffer": True}, "metadata_buffer"),
        ({"deprecated_tag": True}, "deprecated_tag"),
    ],
)
def test_foreign_features_are_refused_precisely(knob: dict, match: str) -> None:
    with pytest.raises(UnsupportedModelError, match=match):
        load_graph_spec(_foreign_model(**knob))


def test_loaded_spec_preserves_structure() -> None:
    spec = load_graph_spec(FIXTURES["model_fix_div_example.tflite"])
    assert [t.dtype for t in spec.tensors] == ["int32", "int32", "int32"]
    assert [op.op for op in spec.ops] == ["DIV"]
    assert spec.ops[0].builtin_options is not None
    assert spec.ops[0].builtin_options.type_code == 29  # DivOptions
    assert spec.inputs == (0, 1)
    assert spec.outputs == (2,)
