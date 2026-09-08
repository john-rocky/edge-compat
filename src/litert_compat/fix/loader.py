"""Load a `.tflite` flatbuffer into the builder's `GraphSpec` for rewriting.

The fix engine mutates a `GraphSpec` and reserializes with `build_tflite` —
so a model is rewritable only if this loader can represent EVERY populated
field faithfully. Real converter output round-trips: tensor quantization
parameters (scale/zero_point/min/max/quantized_dimension), `has_rank`, model
metadata (name + buffer bytes, e.g. `min_runtime_version`), signature_defs
(key + tensor maps), and appended-weight buffers (offset/size pairs pointing
past the flatbuffer — read faithfully, rebuilt inline; refused past the
inline uoffset limit). Anything else it cannot round-trip (custom quantization
`details`, sparsity, variable tensors, multiple subgraphs, custom options,
unregistered builtin-options tables or fields, …) raises
`UnsupportedModelError` with a precise reason: refused, never silently
dropped. The one exception is a fully-default quantization table (no
meaningful fields — semantically void, common in converter output), which is
dropped.

For builder-produced models the loader → `build_tflite` round trip is
byte-identical (tested). Foreign files may renormalize (buffer order, empty
strings) — determinism still holds, byte-identity does not.

vtable slot layouts mirror the public append-only TFLite schema, the same
convention as `parser.reader` (which reads only the linter's subset — this
module reads the full rewrite surface, including buffers and options).
"""

from __future__ import annotations

import flatbuffers
from flatbuffers import number_types as N
from flatbuffers.table import Table

from litert_compat.fix.options_registry import OPTION_LAYOUTS
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
)
from litert_compat.parser.opcodes import builtin_name, tensor_type_name
from litert_compat.parser.reader import FILE_IDENTIFIER, TfliteParseError

# Table slots (voffset = 4 + 2 * slot) we can represent. Any populated slot
# outside these sets is an UnsupportedModelError.
_MODEL_SLOTS = {
    0: "version",
    1: "operator_codes",
    2: "subgraphs",
    3: "description",
    4: "buffers",
    6: "metadata",
    7: "signature_defs",
}
_MODEL_REFUSED = {5: "metadata_buffer"}
_SUBGRAPH_SLOTS = {0: "tensors", 1: "inputs", 2: "outputs", 3: "operators", 4: "name"}
_TENSOR_SLOTS = {
    0: "shape",
    1: "type",
    2: "buffer",
    3: "name",
    4: "quantization",
    7: "shape_signature",
    8: "has_rank",
}
_TENSOR_REFUSED = {
    5: "is_variable",
    6: "sparsity",
    9: "variant_tensors",
}
_QUANT_SLOTS = {0: "min", 1: "max", 2: "scale", 3: "zero_point", 6: "quantized_dimension"}
_QUANT_REFUSED = {4: "details_type", 5: "details"}
_METADATA_SLOTS = {0: "name", 1: "buffer"}
_SIGNATURE_SLOTS = {0: "inputs", 1: "outputs", 2: "signature_key", 4: "subgraph_index"}
_SIGNATURE_REFUSED = {3: "deprecated_tag"}
_TENSOR_MAP_SLOTS = {0: "name", 1: "tensor_index"}
_OPERATOR_SLOTS = {0: "opcode_index", 1: "inputs", 2: "outputs",
                   3: "builtin_options_type", 4: "builtin_options"}
_OPERATOR_REFUSED = {
    5: "custom_options",
    6: "custom_options_format",
    7: "mutating_variable_inputs",
    8: "intermediates",
    9: "large_custom_options_offset",
    10: "large_custom_options_size",
    11: "builtin_options_2_type",
    12: "builtin_options_2",
}
_OPCODE_SLOTS = {0: "deprecated_builtin_code", 1: "custom_code", 2: "version", 3: "builtin_code"}
_BUFFER_SLOTS = {0: "data", 1: "offset", 2: "size"}

# Rebuilding inlines appended-weight buffers; past ~1.9 GB the inline
# flatbuffer itself would overflow uoffset arithmetic — refuse, don't corrupt.
_INLINE_REBUILD_LIMIT = 1_900_000_000

_KIND_FLAGS = {
    "float32": N.Float32Flags,
    "int32": N.Int32Flags,
    "int8": N.Int8Flags,
    "uint8": N.Uint8Flags,
    "bool": N.BoolFlags,
}


class UnsupportedModelError(ValueError):
    """The model contains a feature the rewrite path cannot round-trip
    faithfully. The message states exactly which one and where."""


def _populated_slots(tab: Table) -> set[int]:
    vtable = tab.Pos - tab.Get(N.SOffsetTFlags, tab.Pos)
    vtable_size = tab.Get(N.VOffsetTFlags, vtable)
    slot_count = (vtable_size - 4) // 2
    return {
        slot
        for slot in range(slot_count)
        if tab.Get(N.VOffsetTFlags, vtable + 4 + 2 * slot) != 0
    }


def _check_slots(
    tab: Table, allowed: dict[int, str], refused: dict[int, str], where: str
) -> set[int]:
    populated = _populated_slots(tab)
    for slot in sorted(populated - set(allowed)):
        field = refused.get(slot, f"slot {slot}")
        raise UnsupportedModelError(
            f"{where}: field {field!r} is populated — not representable for "
            "rewriting (refused, never silently dropped)"
        )
    return populated


def _voffset(slot: int) -> int:
    return 4 + 2 * slot


def _string(tab: Table, slot: int) -> str | None:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return None
    raw = tab.String(o + tab.Pos)
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _scalar(tab: Table, slot: int, flags: object, default: int) -> int:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return default
    return int(tab.Get(flags, o + tab.Pos))


def _int32_vector(tab: Table, slot: int) -> tuple[int, ...] | None:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return None
    start = tab.Vector(o)
    return tuple(tab.Get(N.Int32Flags, start + i * 4) for i in range(tab.VectorLen(o)))


def _float32_vector(tab: Table, slot: int) -> tuple[float, ...]:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return ()
    start = tab.Vector(o)
    return tuple(
        float(tab.Get(N.Float32Flags, start + i * 4)) for i in range(tab.VectorLen(o))
    )


def _int64_vector(tab: Table, slot: int) -> tuple[int, ...]:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return ()
    start = tab.Vector(o)
    return tuple(int(tab.Get(N.Int64Flags, start + i * 8)) for i in range(tab.VectorLen(o)))


def _byte_vector(tab: Table, slot: int) -> bytes | None:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return None
    start = tab.Vector(o)
    return bytes(tab.Bytes[start : start + tab.VectorLen(o)])


def _table_vector(tab: Table, slot: int) -> list[Table]:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return []
    return [
        Table(tab.Bytes, tab.Indirect(tab.Vector(o) + i * 4)) for i in range(tab.VectorLen(o))
    ]


def _table(tab: Table, slot: int) -> Table | None:
    o = tab.Offset(_voffset(slot))
    if o == 0:
        return None
    return Table(tab.Bytes, tab.Indirect(o + tab.Pos))


def _read_buffer(buf: Table, index: int, file_data: bytes) -> bytes | None:
    """Inline buffer bytes, or the appended-weight region a populated
    offset/size pair points at (newer converters place weights after the
    flatbuffer). Rebuilds inline either way — semantically identical,
    deterministic."""
    where = f"buffers[{index}]"
    _check_slots(buf, _BUFFER_SLOTS, {}, where)
    inline = _byte_vector(buf, 0)
    offset = _scalar(buf, 1, N.Uint64Flags, 0)
    size = _scalar(buf, 2, N.Uint64Flags, 0)
    if offset == 0 and size == 0:
        return inline
    if inline is not None:
        raise TfliteParseError(f"{where}: both inline data and offset/size are populated")
    if offset == 1:
        raise TfliteParseError(f"{where}: unfinalized placeholder offset 1")
    if offset == 0 or offset + size > len(file_data):
        raise TfliteParseError(
            f"{where}: offset/size ({offset}, {size}) outside the file "
            f"({len(file_data)} bytes)"
        )
    return bytes(file_data[offset : offset + size])


def _read_quantization(tab: Table, where: str) -> QuantizationSpec | None:
    quant = _table(tab, 4)
    if quant is None:
        return None
    _check_slots(quant, _QUANT_SLOTS, _QUANT_REFUSED, f"{where}.quantization")
    spec = QuantizationSpec(
        scale=_float32_vector(quant, 2),
        zero_point=_int64_vector(quant, 3),
        min=_float32_vector(quant, 0),
        max=_float32_vector(quant, 1),
        quantized_dimension=_scalar(quant, 6, N.Int32Flags, 0),
    )
    # A fully-default table is semantically void — dropped on rebuild.
    return None if spec == QuantizationSpec() else spec


def _read_metadata(model: Table, buffers: list[bytes | None]) -> tuple[MetadataSpec, ...]:
    entries = []
    for i, tab in enumerate(_table_vector(model, 6)):
        where = f"metadata[{i}]"
        _check_slots(tab, _METADATA_SLOTS, {}, where)
        buffer_index = _scalar(tab, 1, N.Uint32Flags, 0)
        if buffer_index >= len(buffers):
            raise TfliteParseError(f"{where}: buffer index {buffer_index} out of range")
        entries.append(
            MetadataSpec(name=_string(tab, 0) or "", data=buffers[buffer_index] or b"")
        )
    return tuple(entries)


def _read_tensor_maps(tab: Table, slot: int, tensor_count: int, where: str) -> tuple[
    TensorMapSpec, ...
]:
    maps = []
    for i, map_tab in enumerate(_table_vector(tab, slot)):
        _check_slots(map_tab, _TENSOR_MAP_SLOTS, {}, f"{where}[{i}]")
        tensor_index = _scalar(map_tab, 1, N.Uint32Flags, 0)
        if tensor_index >= tensor_count:
            raise TfliteParseError(
                f"{where}[{i}]: tensor index {tensor_index} out of range"
            )
        maps.append(TensorMapSpec(name=_string(map_tab, 0) or "", tensor_index=tensor_index))
    return tuple(maps)


def _read_signature_defs(model: Table, tensor_count: int) -> tuple[SignatureDefSpec, ...]:
    defs = []
    for i, tab in enumerate(_table_vector(model, 7)):
        where = f"signature_defs[{i}]"
        _check_slots(tab, _SIGNATURE_SLOTS, _SIGNATURE_REFUSED, where)
        subgraph_index = _scalar(tab, 4, N.Uint32Flags, 0)
        if subgraph_index != 0:
            raise UnsupportedModelError(
                f"{where}: subgraph_index {subgraph_index} — only single-subgraph "
                "models are rewritable in v1"
            )
        defs.append(SignatureDefSpec(
            signature_key=_string(tab, 2) or "",
            inputs=_read_tensor_maps(tab, 0, tensor_count, f"{where}.inputs"),
            outputs=_read_tensor_maps(tab, 1, tensor_count, f"{where}.outputs"),
        ))
    return tuple(defs)


def _read_options(op_tab: Table, op: str, where: str) -> BuiltinOptionsSpec | None:
    type_code = _scalar(op_tab, 3, N.Uint8Flags, 0)
    options_tab = _table(op_tab, 4)
    if type_code == 0:
        if options_tab is not None:
            raise UnsupportedModelError(
                f"{where} ({op}): builtin_options table present without a type"
            )
        return None
    if options_tab is None:
        raise UnsupportedModelError(
            f"{where} ({op}): builtin_options type {type_code} set without a table"
        )
    layout = OPTION_LAYOUTS.get(type_code)
    if layout is None:
        raise UnsupportedModelError(
            f"{where} ({op}): builtin-options type {type_code} is outside the "
            "v1 options registry — not representable for rewriting"
        )
    by_slot = {field.slot: field for field in layout}
    fields = []
    for slot in sorted(_populated_slots(options_tab)):
        field = by_slot.get(slot)
        if field is None:
            raise UnsupportedModelError(
                f"{where} ({op}): builtin-options type {type_code} has populated "
                f"field slot {slot} outside the registered layout"
            )
        value = options_tab.Get(_KIND_FLAGS[field.kind],
                                options_tab.Offset(_voffset(slot)) + options_tab.Pos)
        fields.append(OptionsField(slot=slot, kind=field.kind, value=value))
    return BuiltinOptionsSpec(type_code=type_code, fields=tuple(fields))


def _read_opcode(tab: Table, index: int) -> tuple[str, str | None, int]:
    _check_slots(tab, _OPCODE_SLOTS, {}, f"operator_codes[{index}]")
    code = max(_scalar(tab, 0, N.Int8Flags, 0), _scalar(tab, 3, N.Int32Flags, 0))
    name = builtin_name(code)
    if name.startswith("UNKNOWN_BUILTIN_"):
        raise UnsupportedModelError(
            f"operator_codes[{index}]: unknown builtin operator code {code}"
        )
    return name, _string(tab, 1), _scalar(tab, 2, N.Int32Flags, 1)


def _read_tensor(tab: Table, index: int, buffers: list[bytes | None]) -> TensorSpec:
    where = f"tensors[{index}]"
    _check_slots(tab, _TENSOR_SLOTS, _TENSOR_REFUSED, where)
    shape = _int32_vector(tab, 0)
    if shape is None:
        raise UnsupportedModelError(f"{where}: tensor has no shape (unknown rank)")
    dtype = tensor_type_name(_scalar(tab, 1, N.Int8Flags, 0))
    if dtype.startswith("unknown_dtype_"):
        raise UnsupportedModelError(f"{where}: unknown tensor dtype code")
    buffer_index = _scalar(tab, 2, N.Uint32Flags, 0)
    if buffer_index >= len(buffers):
        raise TfliteParseError(f"{where}: buffer index {buffer_index} out of range")
    return TensorSpec(
        name=_string(tab, 3) or "",
        dtype=dtype,
        shape=shape,
        shape_signature=_int32_vector(tab, 7),
        data=buffers[buffer_index] if buffer_index > 0 else None,
        quantization=_read_quantization(tab, where),
        has_rank=bool(_scalar(tab, 8, N.BoolFlags, 0)),
    )


def _read_operator(
    tab: Table, index: int, opcodes: list[tuple[str, str | None, int]], tensor_count: int
) -> OpSpec:
    where = f"operators[{index}]"
    _check_slots(tab, _OPERATOR_SLOTS, _OPERATOR_REFUSED, where)
    opcode_index = _scalar(tab, 0, N.Uint32Flags, 0)
    if opcode_index >= len(opcodes):
        raise TfliteParseError(f"{where}: opcode_index {opcode_index} out of range")
    op, custom_code, version = opcodes[opcode_index]
    inputs = _int32_vector(tab, 1) or ()
    outputs = _int32_vector(tab, 2) or ()
    for tensor_index in inputs:
        if tensor_index == -1:
            continue  # optional input deliberately absent (e.g. bias-free FULLY_CONNECTED)
        if not 0 <= tensor_index < tensor_count:
            raise TfliteParseError(f"{where} ({op}): tensor index {tensor_index} out of range")
    for tensor_index in outputs:
        if not 0 <= tensor_index < tensor_count:
            raise TfliteParseError(f"{where} ({op}): tensor index {tensor_index} out of range")
    if op == "CUSTOM":
        raise UnsupportedModelError(
            f"{where}: custom op {custom_code!r} — custom ops are not rewritable"
        )
    return OpSpec(
        op=op,
        inputs=inputs,
        outputs=outputs,
        custom_code=custom_code,
        version=version,
        builtin_options=_read_options(tab, op, where),
    )


def load_graph_spec(data: bytes) -> GraphSpec:
    """Parse a `.tflite` into a rewritable `GraphSpec`, or raise
    `UnsupportedModelError` (unrepresentable feature) / `TfliteParseError`
    (not a readable flatbuffer)."""
    if len(data) < 8 or data[4:8] != FILE_IDENTIFIER:
        raise TfliteParseError("missing TFLite file identifier — not a .tflite file?")
    try:
        root = flatbuffers.encode.Get(flatbuffers.packer.uoffset, data, 0)
        model = Table(data, root)
        _check_slots(model, _MODEL_SLOTS, _MODEL_REFUSED, "model")
        version = _scalar(model, 0, N.Uint32Flags, 0)
        if version != 3:
            raise UnsupportedModelError(f"model: schema version {version} (expected 3)")

        buffer_tables = _table_vector(model, 4)
        buffers: list[bytes | None] = [
            _read_buffer(buf, i, data) for i, buf in enumerate(buffer_tables)
        ]
        payload = sum(len(b) for b in buffers if b)
        if payload > _INLINE_REBUILD_LIMIT:
            raise UnsupportedModelError(
                f"model: {payload} bytes of buffer payload exceed the inline "
                f"rebuild limit ({_INLINE_REBUILD_LIMIT}) — not representable "
                "for rewriting"
            )

        opcodes = [_read_opcode(t, i) for i, t in enumerate(_table_vector(model, 1))]

        subgraphs = _table_vector(model, 2)
        if len(subgraphs) != 1:
            raise UnsupportedModelError(
                f"model has {len(subgraphs)} subgraphs — only single-subgraph "
                "models are rewritable in v1"
            )
        subgraph = subgraphs[0]
        _check_slots(subgraph, _SUBGRAPH_SLOTS, {}, "subgraph 0")

        tensors = tuple(
            _read_tensor(t, i, buffers) for i, t in enumerate(_table_vector(subgraph, 0))
        )
        ops = tuple(
            _read_operator(t, i, opcodes, len(tensors))
            for i, t in enumerate(_table_vector(subgraph, 3))
        )
        return GraphSpec(
            tensors=tensors,
            ops=ops,
            inputs=_int32_vector(subgraph, 1) or (),
            outputs=_int32_vector(subgraph, 2) or (),
            description=_string(model, 3) or "",
            name=_string(subgraph, 4) or "",
            metadata=_read_metadata(model, buffers),
            signature_defs=_read_signature_defs(model, len(tensors)),
        )
    except (UnsupportedModelError, TfliteParseError):
        raise
    except Exception as exc:  # struct.error, IndexError, UnicodeDecodeError, ...
        raise TfliteParseError(f"malformed TFLite flatbuffer: {exc}") from exc
