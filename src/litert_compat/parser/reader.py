"""Minimal `.tflite` flatbuffer reader.

Reads exactly the tables the linter needs — Model, OperatorCode, SubGraph,
Operator, Tensor — via hand-written accessors on the `flatbuffers` runtime,
using the vtable slot layout of the public TFLite schema (append-only, so slot
offsets are stable). No generated bindings, no LiteRT runtime, no execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import flatbuffers
from flatbuffers import number_types as N
from flatbuffers.table import Table

from litert_compat.parser.opcodes import builtin_name, tensor_type_name

FILE_IDENTIFIER = b"TFL3"
SMALL_CONSTANT_BYTES = 4096

# vtable offsets: field slot n lives at 4 + 2n.
_MODEL_OPERATOR_CODES = 6
_MODEL_SUBGRAPHS = 8
_MODEL_DESCRIPTION = 10
_MODEL_BUFFERS = 12
_BUFFER_DATA = 4
_BUFFER_OFFSET = 6
_BUFFER_SIZE = 8
_OPCODE_DEPRECATED_BUILTIN = 4
_OPCODE_CUSTOM_CODE = 6
_OPCODE_VERSION = 8
_OPCODE_BUILTIN = 10
_SUBGRAPH_TENSORS = 4
_SUBGRAPH_INPUTS = 6
_SUBGRAPH_OUTPUTS = 8
_SUBGRAPH_OPERATORS = 10
_SUBGRAPH_NAME = 12
_OPERATOR_OPCODE_INDEX = 4
_OPERATOR_INPUTS = 6
_OPERATOR_OUTPUTS = 8
_TENSOR_SHAPE = 4
_TENSOR_TYPE = 6
_TENSOR_BUFFER = 8
_TENSOR_NAME = 10
_TENSOR_SHAPE_SIGNATURE = 18


class TfliteParseError(ValueError):
    """The file is not a readable TFLite flatbuffer."""


@dataclass(frozen=True)
class TensorInfo:
    index: int
    name: str | None
    dtype: str
    shape: tuple[int, ...] | None  # None = unknown rank
    shape_signature: tuple[int, ...] | None
    # True when the tensor's buffer carries bytes (inline data, or an appended
    # offset/size pair) — i.e. a weight/constant rather than an activation.
    # Default False keeps older constructors valid.
    is_constant: bool = False
    # The constant's bytes when the payload is small (<= SMALL_CONSTANT_BYTES):
    # enough for shape/paddings/axis vectors, never weights. None otherwise.
    const_data: bytes | None = None

    @property
    def rank(self) -> int | None:
        if self.shape is not None:
            return len(self.shape)
        if self.shape_signature is not None:
            return len(self.shape_signature)
        return None

    @property
    def is_dynamic(self) -> bool:
        """Dynamic dims are -1 in shape_signature (or shape); unknown rank counts."""
        if self.shape_signature is not None:
            return any(dim < 0 for dim in self.shape_signature)
        if self.shape is not None:
            return any(dim < 0 for dim in self.shape)
        return True


@dataclass(frozen=True)
class Node:
    index: int
    op: str  # exact TFLite builtin name, "CUSTOM", or UNKNOWN_BUILTIN_<code>
    custom_code: str | None
    op_version: int
    inputs: tuple[int, ...]  # tensor indices; optional (-1) inputs removed
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class SubgraphGraph:
    index: int
    name: str | None
    tensors: tuple[TensorInfo, ...]
    nodes: tuple[Node, ...]  # flatbuffer order == execution (topological) order
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class ParsedModel:
    description: str | None
    subgraphs: tuple[SubgraphGraph, ...]


def _string(tab: Table, voffset: int) -> str | None:
    o = tab.Offset(voffset)
    if o == 0:
        return None
    raw = tab.String(o + tab.Pos)
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _int32_vector(tab: Table, voffset: int) -> tuple[int, ...] | None:
    o = tab.Offset(voffset)
    if o == 0:
        return None
    start = tab.Vector(o)
    return tuple(
        tab.Get(N.Int32Flags, start + i * 4) for i in range(tab.VectorLen(o))
    )


def _table_vector(tab: Table, voffset: int) -> list[Table]:
    o = tab.Offset(voffset)
    if o == 0:
        return []
    out = []
    for i in range(tab.VectorLen(o)):
        pos = tab.Indirect(tab.Vector(o) + i * 4)
        out.append(Table(tab.Bytes, pos))
    return out


def _scalar(tab: Table, voffset: int, flags: object, default: int) -> int:
    o = tab.Offset(voffset)
    if o == 0:
        return default
    return int(tab.Get(flags, o + tab.Pos))


def _read_opcode(tab: Table) -> tuple[str, str | None, int]:
    """-> (op name, custom_code, op version). Builtin code resolution follows
    upstream schema_utils: max(deprecated_builtin_code, builtin_code)."""
    deprecated = _scalar(tab, _OPCODE_DEPRECATED_BUILTIN, N.Int8Flags, 0)
    builtin = _scalar(tab, _OPCODE_BUILTIN, N.Int32Flags, 0)
    code = max(deprecated, builtin)
    version = _scalar(tab, _OPCODE_VERSION, N.Int32Flags, 1)
    custom_code = _string(tab, _OPCODE_CUSTOM_CODE)
    return builtin_name(code), custom_code, version


def _read_buffer(tab: Table) -> tuple[bool, bytes | None]:
    """-> (has_data, small_payload). A model buffer counts as data-bearing when
    its inline `data` vector is non-empty or it points at appended bytes
    (offset/size, newer converters). Buffer 0 is the empty sentinel by
    convention; an empty buffer at any other index still means 'no data' —
    constness is decided by content, not index. The payload is returned only
    when it is at most SMALL_CONSTANT_BYTES (shape / paddings / axis vectors);
    weights are never copied."""
    o = tab.Offset(_BUFFER_DATA)
    if o != 0 and tab.VectorLen(o) > 0:
        length = tab.VectorLen(o)
        if length <= SMALL_CONSTANT_BYTES:
            start = tab.Vector(o)
            return True, bytes(tab.Bytes[start:start + length])
        return True, None
    size = _scalar(tab, _BUFFER_SIZE, N.Uint64Flags, 0)
    offset = _scalar(tab, _BUFFER_OFFSET, N.Uint64Flags, 0)
    if size > 0 and offset > 1:  # offset 1 is the schema's "unset" placeholder
        if size <= SMALL_CONSTANT_BYTES and offset + size <= len(tab.Bytes):
            return True, bytes(tab.Bytes[offset:offset + size])
        return True, None
    return False, None


def _read_buffer_has_data(tab: Table) -> bool:
    return _read_buffer(tab)[0]


def _read_tensor(
    tab: Table, index: int, buffer_has_data: tuple[bool, ...],
    buffer_payload: tuple[bytes | None, ...] = (),
) -> TensorInfo:
    dtype_code = _scalar(tab, _TENSOR_TYPE, N.Int8Flags, 0)
    buffer_index = _scalar(tab, _TENSOR_BUFFER, N.Uint32Flags, 0)
    is_constant = 0 <= buffer_index < len(buffer_has_data) and buffer_has_data[buffer_index]
    const_data = buffer_payload[buffer_index] if buffer_index < len(buffer_payload) else None
    return TensorInfo(
        index=index,
        name=_string(tab, _TENSOR_NAME),
        dtype=tensor_type_name(dtype_code),
        shape=_int32_vector(tab, _TENSOR_SHAPE),
        shape_signature=_int32_vector(tab, _TENSOR_SHAPE_SIGNATURE),
        is_constant=is_constant,
        const_data=const_data if is_constant else None,
    )


def _read_subgraph(
    tab: Table, index: int, opcodes: list[tuple[str, str | None, int]],
    buffer_has_data: tuple[bool, ...] = (),
    buffer_payload: tuple[bytes | None, ...] = (),
) -> SubgraphGraph:
    tensors = tuple(
        _read_tensor(t, i, buffer_has_data, buffer_payload)
        for i, t in enumerate(_table_vector(tab, _SUBGRAPH_TENSORS))
    )
    nodes = []
    for i, op_tab in enumerate(_table_vector(tab, _SUBGRAPH_OPERATORS)):
        opcode_index = _scalar(op_tab, _OPERATOR_OPCODE_INDEX, N.Uint32Flags, 0)
        if opcode_index >= len(opcodes):
            raise TfliteParseError(
                f"subgraph {index} operator {i}: opcode_index {opcode_index} "
                f"out of range ({len(opcodes)} operator codes)"
            )
        op, custom_code, op_version = opcodes[opcode_index]
        inputs = _int32_vector(op_tab, _OPERATOR_INPUTS) or ()
        outputs = _int32_vector(op_tab, _OPERATOR_OUTPUTS) or ()
        for tensor_index in (*inputs, *outputs):
            if tensor_index >= len(tensors):
                raise TfliteParseError(
                    f"subgraph {index} operator {i} ({op}): tensor index "
                    f"{tensor_index} out of range ({len(tensors)} tensors)"
                )
        nodes.append(
            Node(
                index=i,
                op=op,
                custom_code=custom_code,
                op_version=op_version,
                inputs=tuple(t for t in inputs if t >= 0),
                outputs=tuple(t for t in outputs if t >= 0),
            )
        )
    return SubgraphGraph(
        index=index,
        name=_string(tab, _SUBGRAPH_NAME),
        tensors=tensors,
        nodes=tuple(nodes),
        inputs=_int32_vector(tab, _SUBGRAPH_INPUTS) or (),
        outputs=_int32_vector(tab, _SUBGRAPH_OUTPUTS) or (),
    )


def parse_tflite(data: bytes) -> ParsedModel:
    if len(data) < 8:
        raise TfliteParseError(f"file too small to be a TFLite flatbuffer ({len(data)} bytes)")
    if data[4:8] != FILE_IDENTIFIER:
        raise TfliteParseError(
            f"missing TFLite file identifier {FILE_IDENTIFIER!r} "
            f"(found {bytes(data[4:8])!r}) — not a .tflite file?"
        )
    try:
        root = flatbuffers.encode.Get(flatbuffers.packer.uoffset, data, 0)
        model = Table(data, root)
        opcodes = [_read_opcode(t) for t in _table_vector(model, _MODEL_OPERATOR_CODES)]
        buffers = [_read_buffer(t) for t in _table_vector(model, _MODEL_BUFFERS)]
        buffer_has_data = tuple(has for has, _ in buffers)
        buffer_payload = tuple(payload for _, payload in buffers)
        subgraphs = tuple(
            _read_subgraph(t, i, opcodes, buffer_has_data, buffer_payload)
            for i, t in enumerate(_table_vector(model, _MODEL_SUBGRAPHS))
        )
        description = _string(model, _MODEL_DESCRIPTION)
    except TfliteParseError:
        raise
    except Exception as exc:  # struct.error, IndexError, UnicodeDecodeError, ...
        raise TfliteParseError(f"malformed TFLite flatbuffer: {exc}") from exc
    if not subgraphs:
        raise TfliteParseError("model contains no subgraphs")
    return ParsedModel(description=description, subgraphs=subgraphs)


def parse_tflite_file(path: Path) -> ParsedModel:
    return parse_tflite(path.read_bytes())
