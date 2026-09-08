"""Minimal LiteRT-LM (`.litertlm`) container reader — and a tiny writer for
example fixtures.

Reads exactly what the static linter needs: the container TOC and the bytes of
the embedded TFLite model section. Format learned from the litert-lm SDK's
own tooling (`litert_lm_builder/litertlm_core.py`, `litertlm_peek.py`,
container version 1.5.0), re-implemented here on the `flatbuffers` runtime
with hand-written vtable accessors — the same idiom as `reader.py`. No
litert-lm dependency, no runtime, no execution.

Container layout:

- bytes 0-7    magic ``LITERTLM``
- bytes 8-19   ``<III`` major/minor/patch (little-endian)
- bytes 24-31  ``<Q`` absolute end offset of the header flatbuffer
- bytes 32..header_end   ``LiteRTLMMetaData`` flatbuffer
- section payloads at absolute ``[begin_offset, end_offset)`` per the TOC
  (the SDK block-aligns real files to 16 KiB; readers must trust only the
  TOC offsets, and this one does)

Static-lint honesty note: graph analysis needs only the model section's
operator/tensor tables, so an extracted model is lintable even when the
bundle externalizes weights into a separate ``TFLiteWeights`` section —
buffers are never read by the linter.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flatbuffers import number_types as N
from flatbuffers.table import Table

MAGIC = b"LITERTLM"
_VERSION_OFFSET = 8
_HEADER_END_LOCATION_OFFSET = 24
_HEADER_BEGIN_OFFSET = 32

# AnySectionDataType enum (litertlm_header_schema).
SECTION_TYPE_NAMES = {
    0: "NONE",
    1: "GenericBinaryData",
    2: "Deprecated",
    3: "TFLiteModel",
    4: "SP_Tokenizer",
    5: "LlmMetadataProto",
    6: "HF_Tokenizer_Zlib",
    7: "TFLiteWeights",
}
TFLITE_MODEL = 3

# VData union discriminants for KeyValuePair values.
_VDATA_STRING = 9
_VDATA_SCALARS = {
    1: N.Uint8Flags,
    2: N.Int8Flags,
    3: N.Uint16Flags,
    4: N.Int16Flags,
    5: N.Uint32Flags,
    6: N.Int32Flags,
    7: N.Float32Flags,
    8: N.BoolFlags,
    10: N.Uint64Flags,
    11: N.Int64Flags,
    12: N.Float64Flags,
}


class LitertlmParseError(ValueError):
    """The file is not a readable LiteRT-LM container."""


@dataclass(frozen=True)
class Section:
    """One TOC entry: absolute payload offsets plus its key/value items."""

    index: int
    data_type: int
    data_type_name: str
    begin: int
    end: int
    items: dict[str, Any]

    @property
    def model_type(self) -> str | None:
        value = self.items.get("model_type")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class ParsedBundle:
    version: tuple[int, int, int]
    sections: tuple[Section, ...]


def is_litertlm(data: bytes) -> bool:
    """True when the bytes start with the LiteRT-LM container magic."""
    return data[: len(MAGIC)] == MAGIC


def _string(tab: Table, voffset: int) -> str | None:
    o = tab.Offset(voffset)
    if o == 0:
        return None
    raw = tab.String(o + tab.Pos)
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _scalar(tab: Table, voffset: int, flags: object, default: int) -> int:
    o = tab.Offset(voffset)
    if o == 0:
        return default
    return int(tab.Get(flags, o + tab.Pos))


def _kvp_value(tab: Table) -> Any:
    """KeyValuePair value: ValueType @6 discriminates the Value union @8."""
    value_type = _scalar(tab, 6, N.Uint8Flags, 0)
    o = tab.Offset(8)
    if o == 0:
        return None
    union = Table(bytearray(), 0)
    tab.Union(union, o)
    if value_type == _VDATA_STRING:
        return _string(union, 4)
    flags = _VDATA_SCALARS.get(value_type)
    if flags is None:
        return None
    value = union.Get(flags, union.Pos + union.Offset(4)) if union.Offset(4) else None
    if value is None:
        return None
    return bool(value) if flags is N.BoolFlags else value


def _kvp_items(tab: Table, voffset: int) -> dict[str, Any]:
    o = tab.Offset(voffset)
    if o == 0:
        return {}
    items: dict[str, Any] = {}
    for i in range(tab.VectorLen(o)):
        pos = tab.Indirect(tab.Vector(o) + i * 4)
        kvp = Table(tab.Bytes, pos)
        key = _string(kvp, 4)
        if key is not None:
            items[key] = _kvp_value(kvp)
    return items


def parse_litertlm(data: bytes) -> ParsedBundle:
    """Parse the container header and TOC. Raises LitertlmParseError."""
    if not is_litertlm(data):
        raise LitertlmParseError(
            f"not a LiteRT-LM container (magic {data[:8]!r}, expected {MAGIC!r})"
        )
    if len(data) < _HEADER_BEGIN_OFFSET:
        raise LitertlmParseError(f"truncated container: {len(data)} bytes")
    major, minor, patch = struct.unpack_from("<III", data, _VERSION_OFFSET)
    (header_end,) = struct.unpack_from("<Q", data, _HEADER_END_LOCATION_OFFSET)
    if not _HEADER_BEGIN_OFFSET < header_end <= len(data):
        raise LitertlmParseError(
            f"header end offset {header_end} outside the file ({len(data)} bytes)"
        )
    header = data[_HEADER_BEGIN_OFFSET:header_end]
    try:
        root_pos = struct.unpack_from("<I", header, 0)[0]
        root = Table(bytearray(header), root_pos)
        sections: list[Section] = []
        # LiteRTLMMetaData.SectionMetadata @6 -> SectionMetadata.Objects @4.
        o = root.Offset(6)
        if o != 0:
            meta = Table(root.Bytes, root.Indirect(o + root.Pos))
            vec = meta.Offset(4)
            for i in range(meta.VectorLen(vec) if vec else 0):
                pos = meta.Indirect(meta.Vector(vec) + i * 4)
                obj = Table(meta.Bytes, pos)
                begin = _scalar(obj, 6, N.Uint64Flags, 0)
                end = _scalar(obj, 8, N.Uint64Flags, 0)
                data_type = _scalar(obj, 10, N.Uint8Flags, 0)
                if not 0 <= begin <= end <= len(data):
                    raise LitertlmParseError(
                        f"section {i} offsets [{begin}, {end}) outside the file "
                        f"({len(data)} bytes)"
                    )
                sections.append(
                    Section(
                        index=i,
                        data_type=data_type,
                        data_type_name=SECTION_TYPE_NAMES.get(
                            data_type, f"Unknown({data_type})"
                        ),
                        begin=begin,
                        end=end,
                        items=_kvp_items(obj, 4),
                    )
                )
    except LitertlmParseError:
        raise
    except Exception as exc:  # flatbuffers raises bare exceptions on corrupt input
        raise LitertlmParseError(f"unreadable container header: {exc}") from exc
    return ParsedBundle(version=(major, minor, patch), sections=tuple(sections))


def extract_tflite_model(
    data: bytes, model_type: str | None = None
) -> tuple[bytes, Section]:
    """Bytes of the embedded TFLite model section (plus its TOC entry).

    Exactly one ``TFLiteModel`` section: that one. Multiple: refuse unless
    `model_type` selects exactly one (never guess which bytes are the model).
    """
    bundle = parse_litertlm(data)
    candidates = [s for s in bundle.sections if s.data_type == TFLITE_MODEL]
    if model_type is not None:
        candidates = [s for s in candidates if s.model_type == model_type]
    if not candidates:
        available = [
            f"section {s.index} (model_type={s.model_type or '-'})"
            for s in bundle.sections
            if s.data_type == TFLITE_MODEL
        ]
        raise LitertlmParseError(
            "no TFLiteModel section"
            + (f" with model_type {model_type!r}" if model_type else "")
            + (f"; available: {', '.join(available)}" if available else " in the container")
        )
    if len(candidates) > 1:
        listing = ", ".join(
            f"section {s.index} (model_type={s.model_type or '-'})" for s in candidates
        )
        raise LitertlmParseError(
            f"ambiguous: {len(candidates)} TFLiteModel sections ({listing}); "
            "select one with model_type"
        )
    chosen = candidates[0]
    return data[chosen.begin : chosen.end], chosen


# --- Fixture writer (example bundles only) -----------------------------------
#
# Mirrors the container layout above with a fixed 4096-byte header region so
# output is deterministic (same inputs -> same bytes). Real SDK files align
# sections to 16 KiB blocks; the reader trusts only TOC offsets, so the
# fixture stays tiny instead of mimicking alignment.

_FIXTURE_HEADER_REGION = 4096


def build_bundle(
    sections: list[tuple[int, dict[str, Any], bytes]],
    version: tuple[int, int, int] = (1, 5, 0),
) -> bytes:
    """A minimal container wrapping `sections` = [(data_type, items, payload)]."""
    import flatbuffers

    builder = flatbuffers.Builder(1024)

    offset = _FIXTURE_HEADER_REGION
    placements: list[tuple[int, int]] = []
    for _, _, payload in sections:
        placements.append((offset, offset + len(payload)))
        offset += len(payload)

    section_offsets = []
    for (data_type, items, _), (begin, end) in zip(sections, placements, strict=True):
        kvp_offsets = []
        for key, value in sorted(items.items()):
            if not isinstance(value, str):
                raise LitertlmParseError(
                    f"fixture writer supports string item values only, got {value!r}"
                )
            value_str = builder.CreateString(value)
            builder.StartObject(1)  # StringValue
            builder.PrependUOffsetTRelativeSlot(0, value_str, 0)
            value_obj = builder.EndObject()
            key_str = builder.CreateString(key)
            builder.StartObject(3)  # KeyValuePair
            builder.PrependUOffsetTRelativeSlot(0, key_str, 0)
            builder.PrependUint8Slot(1, _VDATA_STRING, 0)
            builder.PrependUOffsetTRelativeSlot(2, value_obj, 0)
            kvp_offsets.append(builder.EndObject())
        items_vec = 0
        if kvp_offsets:
            builder.StartVector(4, len(kvp_offsets), 4)
            for kvp in reversed(kvp_offsets):
                builder.PrependUOffsetTRelative(kvp)
            items_vec = builder.EndVector()
        builder.StartObject(4)  # SectionObject
        if items_vec:
            builder.PrependUOffsetTRelativeSlot(0, items_vec, 0)
        builder.PrependUint64Slot(1, begin, 0)
        builder.PrependUint64Slot(2, end, 0)
        builder.PrependUint8Slot(3, data_type, 0)
        section_offsets.append(builder.EndObject())

    builder.StartVector(4, len(section_offsets), 4)
    for so in reversed(section_offsets):
        builder.PrependUOffsetTRelative(so)
    objects_vec = builder.EndVector()
    builder.StartObject(1)  # SectionMetadata
    builder.PrependUOffsetTRelativeSlot(0, objects_vec, 0)
    section_metadata = builder.EndObject()

    builder.StartObject(2)  # LiteRTLMMetaData
    builder.PrependUOffsetTRelativeSlot(1, section_metadata, 0)
    builder.Finish(builder.EndObject())
    header = bytes(builder.Output())

    header_end = _HEADER_BEGIN_OFFSET + len(header)
    if header_end > _FIXTURE_HEADER_REGION:
        raise LitertlmParseError(
            f"fixture header {header_end} bytes exceeds the "
            f"{_FIXTURE_HEADER_REGION}-byte fixture region"
        )
    out = bytearray()
    out += MAGIC
    out += struct.pack("<III", *version)
    out += b"\x00" * (_HEADER_END_LOCATION_OFFSET - len(out))
    out += struct.pack("<Q", header_end)
    out += header
    out += b"\x00" * (_FIXTURE_HEADER_REGION - len(out))
    for (_, _, payload), (begin, _) in zip(sections, placements, strict=True):
        assert len(out) == begin
        out += payload
    return bytes(out)


def build_example_bundle(examples_dir: Path) -> bytes:
    """The committed example bundle: wraps model_clean_example.tflite plus a
    generic section, so TOC iteration and section selection are exercised."""
    model = (examples_dir / "model_clean_example.tflite").read_bytes()
    return build_bundle(
        [
            (1, {"note": "example-provenance filler section"}, b"example-bytes"),
            (TFLITE_MODEL, {"model_type": "tf_lite_prefill_decode"}, model),
        ]
    )


if __name__ == "__main__":
    examples = Path(__file__).resolve().parents[3] / "data" / "examples"
    target = examples / "model_bundle_example.litertlm"
    target.write_bytes(build_example_bundle(examples))
    print(f"wrote {target}")
