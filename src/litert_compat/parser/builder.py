"""Deterministic synthetic `.tflite` builder for test fixtures and probes.

Hand-constructs minimal single-subgraph flatbuffers with the `flatbuffers`
Builder — same vtable slot layout `reader.py` reads. Tensors carry no data by
default (they reference the empty sentinel buffer 0); a `TensorSpec` with
`data` becomes a constant tensor with a real 16-byte-aligned buffer, and an
`OpSpec` may carry builtin options — both exist so Phase 8 probe fixtures are
executable by the real LiteRT runtime, not just readable by the static linter.
Same specs → same bytes.

The rewrite surface (`fix/loader.py` → mutate → rebuild) additionally uses
tensor `quantization`, model `metadata` (each entry materializes its own
buffer after the constant-tensor buffers), and model `signature_defs` — all
empty by default, in which case the emitted bytes are unchanged from the
pre-fidelity builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import flatbuffers

from litert_compat.parser.opcodes import BUILTIN_NAME_TO_CODE, DTYPE_TO_TENSOR_TYPE

_PLACEHOLDER_FOR_GREATER_OP_CODES = 127


@dataclass(frozen=True)
class OptionsField:
    """One scalar field of a builtin-options table (vtable slot + typed value)."""

    slot: int
    kind: str  # "float32" | "int32" | "int8" | "uint8" | "bool"
    value: float | int | bool
    default: float | int | bool = 0


@dataclass(frozen=True)
class BuiltinOptionsSpec:
    """Operator builtin options: the BuiltinOptions union type code plus the
    option table's scalar fields. An empty `fields` writes an empty table
    (every field at its schema default), which is what converters emit for
    ops like ADD with no fused activation."""

    type_code: int  # BuiltinOptions union enum value from schema.fbs
    fields: tuple[OptionsField, ...] = ()


@dataclass(frozen=True)
class QuantizationSpec:
    """TFLite `QuantizationParameters`: the four parallel vectors plus
    `quantized_dimension` (per-channel axis). The `details` union (custom
    quantization) is not representable. An all-default spec is semantically
    void and is not written — the loader never produces one."""

    scale: tuple[float, ...] = ()
    zero_point: tuple[int, ...] = ()
    min: tuple[float, ...] = ()
    max: tuple[float, ...] = ()
    quantized_dimension: int = 0


@dataclass(frozen=True)
class MetadataSpec:
    """One model `metadata` entry; `data` is the referenced buffer's bytes
    (e.g. name `min_runtime_version` → an ASCII version string). Buffers are
    materialized per entry on build."""

    name: str
    data: bytes


@dataclass(frozen=True)
class TensorMapSpec:
    """One `SignatureDef` input/output: exported name → subgraph tensor."""

    name: str
    tensor_index: int


@dataclass(frozen=True)
class SignatureDefSpec:
    """One model `signature_defs` entry. `subgraph_index` is always 0 by
    construction — this builder emits single-subgraph models only."""

    signature_key: str
    inputs: tuple[TensorMapSpec, ...]
    outputs: tuple[TensorMapSpec, ...]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str  # matrix vocabulary: float32, int8, ...
    shape: tuple[int, ...]
    shape_signature: tuple[int, ...] | None = None  # -1 marks a dynamic dim
    data: bytes | None = None  # constant tensor data; None = empty sentinel buffer
    quantization: QuantizationSpec | None = None
    has_rank: bool = False  # newer converters mark ranked shapes explicitly


@dataclass(frozen=True)
class OpSpec:
    op: str  # exact TFLite builtin name; "CUSTOM" requires custom_code
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    custom_code: str | None = None
    version: int = 1
    builtin_options: BuiltinOptionsSpec | None = None


@dataclass(frozen=True)
class GraphSpec:
    tensors: tuple[TensorSpec, ...]
    ops: tuple[OpSpec, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    # Pre-rename spelling on purpose: the fixture bytes are pinned by tests.
    description: str = "synthetic example fixture (litert-compat)"
    name: str = "main"
    metadata: tuple[MetadataSpec, ...] = ()
    signature_defs: tuple[SignatureDefSpec, ...] = ()


def _int32_vector(b: flatbuffers.Builder, values: tuple[int, ...]) -> int:
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependInt32(v)
    return b.EndVector()


def _offset_vector(b: flatbuffers.Builder, offsets: list[int]) -> int:
    b.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        b.PrependUOffsetTRelative(off)
    return b.EndVector()


def _float32_vector(b: flatbuffers.Builder, values: tuple[float, ...]) -> int:
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependFloat32(v)
    return b.EndVector()


def _int64_vector(b: flatbuffers.Builder, values: tuple[int, ...]) -> int:
    b.StartVector(8, len(values), 8)
    for v in reversed(values):
        b.PrependInt64(v)
    return b.EndVector()


def _build_quantization(b: flatbuffers.Builder, quant: QuantizationSpec) -> int:
    # Populated-but-empty vectors renormalize to absent slots (foreign files
    # only; this builder's loader never produces them).
    min_v = _float32_vector(b, quant.min) if quant.min else None
    max_v = _float32_vector(b, quant.max) if quant.max else None
    scale_v = _float32_vector(b, quant.scale) if quant.scale else None
    zero_v = _int64_vector(b, quant.zero_point) if quant.zero_point else None
    b.StartObject(7)
    if min_v is not None:
        b.PrependUOffsetTRelativeSlot(0, min_v, 0)
    if max_v is not None:
        b.PrependUOffsetTRelativeSlot(1, max_v, 0)
    if scale_v is not None:
        b.PrependUOffsetTRelativeSlot(2, scale_v, 0)
    if zero_v is not None:
        b.PrependUOffsetTRelativeSlot(3, zero_v, 0)
    b.PrependInt32Slot(6, quant.quantized_dimension, 0)
    return b.EndObject()


def _build_tensor(b: flatbuffers.Builder, spec: TensorSpec, buffer_index: int) -> int:
    if spec.dtype not in DTYPE_TO_TENSOR_TYPE:
        raise ValueError(f"tensor {spec.name}: unknown dtype {spec.dtype!r}")
    name = b.CreateString(spec.name)
    shape = _int32_vector(b, spec.shape)
    signature = (
        _int32_vector(b, spec.shape_signature) if spec.shape_signature is not None else None
    )
    quant = (
        _build_quantization(b, spec.quantization)
        if spec.quantization is not None and spec.quantization != QuantizationSpec()
        else None
    )
    b.StartObject(9)  # slots 0..8; unwritten trailing slots are trimmed
    b.PrependUOffsetTRelativeSlot(0, shape, 0)
    b.PrependInt8Slot(1, DTYPE_TO_TENSOR_TYPE[spec.dtype], 0)
    b.PrependUint32Slot(2, buffer_index, 0)  # 0: the empty sentinel
    b.PrependUOffsetTRelativeSlot(3, name, 0)
    if quant is not None:
        b.PrependUOffsetTRelativeSlot(4, quant, 0)
    if signature is not None:
        b.PrependUOffsetTRelativeSlot(7, signature, 0)
    if spec.has_rank:
        b.PrependBoolSlot(8, True, 0)
    return b.EndObject()


_OPTIONS_SLOT_WRITERS = {
    "float32": flatbuffers.Builder.PrependFloat32Slot,
    "int32": flatbuffers.Builder.PrependInt32Slot,
    "int8": flatbuffers.Builder.PrependInt8Slot,
    "uint8": flatbuffers.Builder.PrependUint8Slot,
    "bool": flatbuffers.Builder.PrependBoolSlot,
}


def _build_options(b: flatbuffers.Builder, spec: BuiltinOptionsSpec) -> int:
    for field in spec.fields:
        if field.kind not in _OPTIONS_SLOT_WRITERS:
            raise ValueError(f"unknown options field kind {field.kind!r}")
    slot_count = max((field.slot for field in spec.fields), default=-1) + 1
    b.StartObject(slot_count)
    for field in spec.fields:
        _OPTIONS_SLOT_WRITERS[field.kind](b, field.slot, field.value, field.default)
    return b.EndObject()


def _build_data_buffer(b: flatbuffers.Builder, data: bytes) -> int:
    # Buffer.data is [ubyte] with force_align 16 in the TFLite schema.
    # Bulk copy instead of per-byte Prepend: identical bytes, and it keeps
    # real-model weight buffers (tens of MB) from taking minutes.
    b.StartVector(1, len(data), 16)
    b.head = b.head - len(data)
    b.Bytes[b.head : b.head + len(data)] = data
    data_vector = b.EndVector()
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, data_vector, 0)
    return b.EndObject()


def _build_tensor_maps(
    b: flatbuffers.Builder, maps: tuple[TensorMapSpec, ...], tensor_count: int, where: str
) -> int:
    offsets = []
    for m in maps:
        if not 0 <= m.tensor_index < tensor_count:
            raise ValueError(
                f"signature {where!r}: tensor index {m.tensor_index} out of range"
            )
        name = b.CreateString(m.name)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, name, 0)
        b.PrependUint32Slot(1, m.tensor_index, 0)
        offsets.append(b.EndObject())
    return _offset_vector(b, offsets)


def _build_signature_def(b: flatbuffers.Builder, sig: SignatureDefSpec, tensor_count: int) -> int:
    inputs = _build_tensor_maps(b, sig.inputs, tensor_count, sig.signature_key)
    outputs = _build_tensor_maps(b, sig.outputs, tensor_count, sig.signature_key)
    key = b.CreateString(sig.signature_key)
    b.StartObject(5)  # slot 3 (deprecated_tag) never written; subgraph_index stays 0
    b.PrependUOffsetTRelativeSlot(0, inputs, 0)
    b.PrependUOffsetTRelativeSlot(1, outputs, 0)
    b.PrependUOffsetTRelativeSlot(2, key, 0)
    return b.EndObject()


def _build_operator_code(
    b: flatbuffers.Builder, op: str, custom_code: str | None, version: int
) -> int:
    if op == "CUSTOM" and not custom_code:
        raise ValueError("CUSTOM op requires custom_code")
    if op not in BUILTIN_NAME_TO_CODE:
        raise ValueError(f"unknown builtin operator name {op!r}")
    code = BUILTIN_NAME_TO_CODE[op]
    custom = b.CreateString(custom_code) if custom_code else None
    b.StartObject(4)
    b.PrependInt8Slot(0, min(code, _PLACEHOLDER_FOR_GREATER_OP_CODES), 0)
    if custom is not None:
        b.PrependUOffsetTRelativeSlot(1, custom, 0)
    b.PrependInt32Slot(2, version, 1)
    b.PrependInt32Slot(3, code, 0)
    return b.EndObject()


def build_tflite(graph: GraphSpec) -> bytes:
    """Build a single-subgraph .tflite flatbuffer. Deterministic."""
    for op in graph.ops:
        for tensor_index in op.inputs:
            if tensor_index == -1:
                continue  # optional input deliberately absent
            if not 0 <= tensor_index < len(graph.tensors):
                raise ValueError(f"op {op.op}: tensor index {tensor_index} out of range")
        for tensor_index in op.outputs:
            if not 0 <= tensor_index < len(graph.tensors):
                raise ValueError(f"op {op.op}: tensor index {tensor_index} out of range")

    b = flatbuffers.Builder(1024)

    # Operator codes, deduped by (op, custom_code, version), first appearance wins.
    code_keys: list[tuple[str, str | None, int]] = []
    code_index: dict[tuple[str, str | None, int], int] = {}
    for op in graph.ops:
        key = (op.op, op.custom_code, op.version)
        if key not in code_index:
            code_index[key] = len(code_keys)
            code_keys.append(key)
    opcode_offsets = [_build_operator_code(b, *key) for key in code_keys]
    opcode_vector = _offset_vector(b, opcode_offsets)

    # Constant tensors get real buffers 1..n, in tensor order; the rest share
    # the empty sentinel buffer 0.
    data_tensor_positions = [i for i, spec in enumerate(graph.tensors) if spec.data is not None]
    buffer_index = {pos: 1 + k for k, pos in enumerate(data_tensor_positions)}

    tensor_offsets = [
        _build_tensor(b, spec, buffer_index.get(i, 0)) for i, spec in enumerate(graph.tensors)
    ]
    tensor_vector = _offset_vector(b, tensor_offsets)

    operator_offsets = []
    for op in graph.ops:
        inputs = _int32_vector(b, op.inputs)
        outputs = _int32_vector(b, op.outputs)
        options = _build_options(b, op.builtin_options) if op.builtin_options else None
        b.StartObject(9)
        b.PrependUint32Slot(0, code_index[(op.op, op.custom_code, op.version)], 0)
        b.PrependUOffsetTRelativeSlot(1, inputs, 0)
        b.PrependUOffsetTRelativeSlot(2, outputs, 0)
        if options is not None:
            b.PrependUint8Slot(3, op.builtin_options.type_code, 0)
            b.PrependUOffsetTRelativeSlot(4, options, 0)
        operator_offsets.append(b.EndObject())
    operator_vector = _offset_vector(b, operator_offsets)

    subgraph_inputs = _int32_vector(b, graph.inputs)
    subgraph_outputs = _int32_vector(b, graph.outputs)
    subgraph_name = b.CreateString(graph.name)
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, tensor_vector, 0)
    b.PrependUOffsetTRelativeSlot(1, subgraph_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraph_outputs, 0)
    b.PrependUOffsetTRelativeSlot(3, operator_vector, 0)
    b.PrependUOffsetTRelativeSlot(4, subgraph_name, 0)
    subgraph = b.EndObject()
    subgraph_vector = _offset_vector(b, [subgraph])

    # Buffer 0: the empty sentinel; then one real buffer per constant tensor,
    # then one per metadata entry.
    b.StartObject(1)
    empty_buffer = b.EndObject()
    data_buffers = [
        _build_data_buffer(b, graph.tensors[pos].data or b"") for pos in data_tensor_positions
    ]
    metadata_buffers = [_build_data_buffer(b, m.data) for m in graph.metadata]
    buffer_vector = _offset_vector(b, [empty_buffer, *data_buffers, *metadata_buffers])

    metadata_vector = None
    if graph.metadata:
        first_metadata_buffer = 1 + len(data_tensor_positions)
        metadata_offsets = []
        for k, m in enumerate(graph.metadata):
            name = b.CreateString(m.name)
            b.StartObject(2)
            b.PrependUOffsetTRelativeSlot(0, name, 0)
            b.PrependUint32Slot(1, first_metadata_buffer + k, 0)
            metadata_offsets.append(b.EndObject())
        metadata_vector = _offset_vector(b, metadata_offsets)

    signature_vector = None
    if graph.signature_defs:
        signature_vector = _offset_vector(
            b,
            [_build_signature_def(b, sig, len(graph.tensors)) for sig in graph.signature_defs],
        )

    description = b.CreateString(graph.description)
    b.StartObject(8)
    b.PrependUint32Slot(0, 3, 0)  # schema version
    b.PrependUOffsetTRelativeSlot(1, opcode_vector, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraph_vector, 0)
    b.PrependUOffsetTRelativeSlot(3, description, 0)
    b.PrependUOffsetTRelativeSlot(4, buffer_vector, 0)
    if metadata_vector is not None:
        b.PrependUOffsetTRelativeSlot(6, metadata_vector, 0)
    if signature_vector is not None:
        b.PrependUOffsetTRelativeSlot(7, signature_vector, 0)
    model = b.EndObject()

    b.Finish(model, file_identifier=b"TFL3")
    return bytes(b.Output())
