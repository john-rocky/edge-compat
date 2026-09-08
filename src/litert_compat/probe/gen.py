"""Probe fixture generator: signature → minimal single-op `.tflite`.

Deterministic: the same signature always yields the same fixture id, path, and
bytes. Every generated fixture is round-trip verified — parsed back and
re-classified to prove it exercises exactly the requested (op, dtypes,
shape_meta) — so a probe can never silently measure a different signature
than the one it upgrades.

Micrograph honesty rule (spec §8.1): a probe measures the op in isolation;
full-graph behavior can differ. Probe-derived entries therefore record
`evidence.source_model: "probe:<fixture-id>"`, keeping them distinguishable
from full-model measurements.

The template registry covers option-free unary elementwise ops, binary
elementwise ops, SOFTMAX (beta=1.0), static RESHAPE, and — graduated from the
Phase 8 deferred item — the weighted/structured fp32-activation families:
convolutions (CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV), FULLY_CONNECTED,
pools, BATCH_MATMUL, CONCATENATION, TRANSPOSE, SLICE, PAD, the reducers
(SUM/MEAN/REDUCE_MAX/REDUCE_MIN), LEAKY_RELU, CAST, comparisons
(GREATER/EQUAL), SELECT/SELECT_V2, the resizes, and EMBEDDING_LOOKUP.
Graduated 2026-08-12 (the quantized-weight deferral): DEQUANTIZE (float16 →
float32, the dominant zoo form) and int8-activation forms with canonical
power-of-two quantization for CONV_2D, DEPTHWISE_CONV_2D (per-channel
filters), FULLY_CONNECTED, the pools, PAD, TRANSPOSE, RESHAPE, and the
quantizable binary elementwise ops (ADD/MUL/SUB/MAXIMUM/MINIMUM/
SQUARED_DIFFERENCE). Signatures without a template are *unprobeable* and
surfaced as remaining work — option-constrained forms stay deferred.

A lookup signature names output dtypes + rank (the classification contract);
a fixture is one concrete, deterministic instantiation of it — kernel sizes,
strides, and constant weights are fixed by the template and readable from the
fixture itself, so what was measured is never ambiguous.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from litert_compat.lint.classify import node_signature
from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    OpSpec,
    OptionsField,
    QuantizationSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.parser.opcodes import DTYPE_TO_TENSOR_TYPE
from litert_compat.parser.reader import parse_tflite
from litert_compat.probe.signatures import ProbeSignature, UnprobeableSignatureError

# Option-free unary elementwise ops: out = f(in), same shape and dtype, no
# builtin options required by the runtime (the Phase 5 sweep ran NEG/LOGISTIC/
# ABS/SQRT/TANH fixtures on the real runtime without options).
UNARY_ELEMENTWISE = frozenset(
    {
        "ABS",
        "CEIL",
        "COS",
        "ELU",
        "EXP",
        "FLOOR",
        "HARD_SWISH",
        "LOG",
        "LOGISTIC",
        "NEG",
        "RELU",
        "RELU6",
        "RELU_N1_TO_1",
        "ROUND",
        "RSQRT",
        "SIN",
        "RELU_0_TO_1",
        "SQRT",
        "SQUARE",
        "TANH",
    }
)

# Binary elementwise ops → BuiltinOptions union type code (schema.fbs). Each
# gets an empty options table of the right type: every field at its schema
# default (fused activation NONE), matching what converters emit.
BINARY_ELEMENTWISE: dict[str, int] = {
    "ADD": 11,
    "DIV": 29,
    "FLOOR_DIV": 65,
    "FLOOR_MOD": 72,
    "MAXIMUM": 39,
    "MINIMUM": 39,
    "MUL": 21,
    "POW": 56,
    "SQUARED_DIFFERENCE": 76,
    "SUB": 28,
}

_SOFTMAX_OPTIONS = BuiltinOptionsSpec(
    type_code=9,  # SoftmaxOptions
    fields=(OptionsField(slot=0, kind="float32", value=1.0),),  # beta
)

# BuiltinOptions union type codes (schema.fbs) for the weighted/structured
# templates. Option choices are the minimal executable form: SAME padding,
# stride 1, 2x2 kernels, fused activation NONE, dilation at schema default 1.
_CONV_2D_OPTIONS = BuiltinOptionsSpec(
    type_code=1,  # Conv2DOptions: stride_w=1, stride_h=1
    fields=(
        OptionsField(slot=1, kind="int32", value=1),
        OptionsField(slot=2, kind="int32", value=1),
    ),
)
_DEPTHWISE_CONV_2D_OPTIONS = BuiltinOptionsSpec(
    type_code=2,  # DepthwiseConv2DOptions: stride 1x1, depth_multiplier=1
    fields=(
        OptionsField(slot=1, kind="int32", value=1),
        OptionsField(slot=2, kind="int32", value=1),
        OptionsField(slot=3, kind="int32", value=1),
    ),
)
_POOL_2D_OPTIONS = BuiltinOptionsSpec(
    type_code=5,  # Pool2DOptions: stride 1x1, filter 2x2
    fields=(
        OptionsField(slot=1, kind="int32", value=1),
        OptionsField(slot=2, kind="int32", value=1),
        OptionsField(slot=3, kind="int32", value=2),
        OptionsField(slot=4, kind="int32", value=2),
    ),
)
_FC_OPTIONS = BuiltinOptionsSpec(type_code=8)  # FullyConnectedOptions, all defaults
_FC_KEEP_DIMS_OPTIONS = BuiltinOptionsSpec(
    type_code=8,  # keep_num_dims=true preserves the batch dims (op version 5)
    fields=(OptionsField(slot=2, kind="bool", value=True),),
)
_TRANSPOSE_CONV_OPTIONS = BuiltinOptionsSpec(
    type_code=49,  # TransposeConvOptions: stride 1x1
    fields=(
        OptionsField(slot=1, kind="int32", value=1),
        OptionsField(slot=2, kind="int32", value=1),
    ),
)
_REDUCER_OPTIONS = BuiltinOptionsSpec(
    type_code=27,  # ReducerOptions: keep_dims=true (output rank == input rank)
    fields=(OptionsField(slot=0, kind="bool", value=True),),
)
_LEAKY_RELU_OPTIONS = BuiltinOptionsSpec(
    type_code=75,  # LeakyReluOptions: alpha
    fields=(OptionsField(slot=0, kind="float32", value=0.2),),
)
# Ops whose options table is empty (every field at its schema default).
_CONCATENATION_TYPE = 10
_RESIZE_BILINEAR_OPTIONS = BuiltinOptionsSpec(type_code=15)
_PAD_OPTIONS = BuiltinOptionsSpec(type_code=22)
_TRANSPOSE_OPTIONS = BuiltinOptionsSpec(type_code=26)
_CAST_OPTIONS = BuiltinOptionsSpec(type_code=37)  # kernel reads tensor dtypes
_GREATER_OPTIONS = BuiltinOptionsSpec(type_code=44)
_SELECT_OPTIONS = BuiltinOptionsSpec(type_code=47)
_SLICE_OPTIONS = BuiltinOptionsSpec(type_code=48)
_EQUAL_OPTIONS = BuiltinOptionsSpec(type_code=53)
# The three codes below were corrected 2026-08-12 against real converter
# output (zoo census, DECISIONS #155): the old values (71/95/98) sat on
# other union members. All three tables are emitted empty, and the runtime's
# checked union accessor fell back to identical defaults, so prior probe
# measurements were semantically as-declared.
_RESIZE_NEAREST_OPTIONS = BuiltinOptionsSpec(type_code=74)
_SELECT_V2_OPTIONS = BuiltinOptionsSpec(type_code=98)
_BATCH_MATMUL_OPTIONS = BuiltinOptionsSpec(type_code=101)

_REDUCERS = frozenset({"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN"})
_COMPARISONS = {"GREATER": _GREATER_OPTIONS, "EQUAL": _EQUAL_OPTIONS}
_SELECTS = {"SELECT": _SELECT_OPTIONS, "SELECT_V2": _SELECT_V2_OPTIONS}
_RESIZES = {
    "RESIZE_BILINEAR": _RESIZE_BILINEAR_OPTIONS,
    "RESIZE_NEAREST_NEIGHBOR": _RESIZE_NEAREST_OPTIONS,
}
_RANK4_ONLY = frozenset(
    {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "TRANSPOSE_CONV",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
    }
)
_WEIGHTED_OPS = (
    _RANK4_ONLY
    | _REDUCERS
    | set(_COMPARISONS)
    | set(_SELECTS)
    | {
        "FULLY_CONNECTED",
        "BATCH_MATMUL",
        "CONCATENATION",
        "TRANSPOSE",
        "SLICE",
        "PAD",
        "LEAKY_RELU",
        "CAST",
        "EMBEDDING_LOOKUP",
    }
)
# The feature-input depth every weighted template contracts on (Cin / K).
_PROBE_DEPTH = 4

# Canonical int8 quantization for probe fixtures: exact powers of two keep
# requantization multipliers well-conditioned, and XNNPack's strict check
# bias_scale == input_scale * filter_scale holds exactly (0.5 * 0.25 = 0.125).
_INT8_ACT_QPARAMS = QuantizationSpec(scale=(0.5,), zero_point=(0,))
_INT8_WEIGHT_SCALE = 0.25
_INT8_BIAS_SCALE = 0.125  # input_scale * weight_scale

# Binary elementwise ops with real int8 quantized kernels (version 2). The
# rest of BINARY_ELEMENTWISE has no int8 kernel — int8 signatures there are
# unprobeable with a precise reason, never built-to-crash.
_INT8_BINARY = frozenset(
    {"ADD", "MUL", "SUB", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE"}
)
# Unary elementwise ops whose int8 form takes the canonical qparams
# unchanged (activation pass-through / no fixed-scale contract). TANH,
# LOGISTIC, SOFTMAX and friends pin their output scale in the kernel
# (1/128, 1/256, …) — those forms stay untemplated: an int8 unary fixture
# without its kernel's quantization contract would measure a graph no
# converter produces.
_INT8_UNARY = frozenset({"HARD_SWISH", "RELU", "RELU6", "RELU_N1_TO_1", "RELU_0_TO_1"})
# Weighted/structured families with an int8-activation template.
_INT8_WEIGHTED = frozenset(
    {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "PAD",
        "TRANSPOSE",
    }
)


def _per_channel_qparams(channels: int, axis: int, scale: float) -> QuantizationSpec:
    return QuantizationSpec(
        scale=(scale,) * channels,
        zero_point=(0,) * channels,
        quantized_dimension=axis,
    )

_PROBE_DESCRIPTION = "single-op probe fixture (edge-compat)"

# Deterministic dims for a given output rank; rank 4 exercises (2, 3, 4, 8).
_BASE_DIMS = (2, 3, 4, 8)


@dataclass(frozen=True)
class ProbeFixture:
    signature: ProbeSignature
    fixture_id: str
    path: Path


def _shape_for_rank(rank: int) -> tuple[int, ...]:
    if rank <= 4:
        return _BASE_DIMS[4 - rank :] if rank > 0 else ()
    return (1,) * (rank - 4) + _BASE_DIMS


def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _const_floats(count: int, key: str) -> bytes:
    """Deterministic float32 constant data in [-0.5, 0.5), keyed by `key`.
    Same signature → same weights → byte-identical fixture (the probe-gen
    contract); small magnitudes keep accumulations well-conditioned."""
    out = bytearray()
    counter = 0
    while len(out) < count * 4:
        block = hashlib.sha256(f"{key}:{counter}".encode()).digest()
        for i in range(0, 32, 4):
            value = int.from_bytes(block[i : i + 4], "big") / 2**32 - 0.5
            out += struct.pack("<f", value)
        counter += 1
    return bytes(out[: count * 4])


def _int32_data(values: tuple[int, ...]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def _const_int8s(count: int, key: str) -> bytes:
    """Deterministic int8 weight data in [-5, 5], keyed by `key` — the int8
    counterpart of `_const_floats` (same determinism contract)."""
    out = bytearray()
    counter = 0
    while len(out) < count:
        block = hashlib.sha256(f"{key}:{counter}".encode()).digest()
        out += bytes(((b % 11) - 5) % 256 for b in block)
        counter += 1
    return bytes(out[:count])


def _const_int32s(count: int, key: str) -> bytes:
    """Deterministic int32 bias data in [-4, 4], keyed by `key`."""
    values: list[int] = []
    counter = 0
    while len(values) < count:
        block = hashlib.sha256(f"{key}:{counter}".encode()).digest()
        values.extend((b % 9) - 4 for b in block)
        counter += 1
    return struct.pack(f"<{count}i", *values[:count])


def build_probe_graph(sig: ProbeSignature) -> GraphSpec:
    """Build the single-op graph for a signature, or raise UnprobeableSignatureError."""
    if not sig.dtypes:
        raise UnprobeableSignatureError("signature carries no dtypes")
    if len(sig.dtypes) > 1:
        raise UnprobeableSignatureError(
            "multi-dtype signature has no single-op template"
        )
    dtype = sig.dtypes[0]
    if dtype not in DTYPE_TO_TENSOR_TYPE:
        raise UnprobeableSignatureError(f"unknown dtype {dtype!r}")

    meta = sig.shape_meta
    rank = meta.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise UnprobeableSignatureError("signature has unknown output rank")
    dynamic = bool(meta.get("dynamic_shape", False))
    if dynamic and rank < 1:
        raise UnprobeableSignatureError("dynamic shape requires rank >= 1")

    shape = _shape_for_rank(rank)
    signature_dims = (-1, *shape[1:]) if dynamic else None

    if sig.op in UNARY_ELEMENTWISE or sig.op == "SOFTMAX":
        if sig.op == "SOFTMAX" and rank < 1:
            raise UnprobeableSignatureError("SOFTMAX requires rank >= 1")
        quant = None
        if dtype == "int8":
            if sig.op not in _INT8_UNARY:
                raise UnprobeableSignatureError(
                    f"int8 {sig.op} pins its quantization contract in the kernel "
                    "(fixed output scale) — no faithful template"
                )
            quant = _INT8_ACT_QPARAMS
        options = _SOFTMAX_OPTIONS if sig.op == "SOFTMAX" else None
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, shape, shape_signature=signature_dims,
                           quantization=quant),
                TensorSpec("output", dtype, shape, shape_signature=signature_dims,
                           quantization=quant),
            ),
            ops=(OpSpec(sig.op, (0,), (1,), builtin_options=options),),
            inputs=(0,),
            outputs=(1,),
            description=_PROBE_DESCRIPTION,
        )

    if sig.op in BINARY_ELEMENTWISE:
        quant = None
        version = 1
        if dtype == "int8":
            if sig.op not in _INT8_BINARY:
                raise UnprobeableSignatureError(
                    f"{sig.op} has no int8 quantized kernel — int8 signature "
                    "is unprobeable"
                )
            quant = _INT8_ACT_QPARAMS
            version = 2
        options = BuiltinOptionsSpec(type_code=BINARY_ELEMENTWISE[sig.op])
        return GraphSpec(
            tensors=(
                TensorSpec("input_a", dtype, shape, shape_signature=signature_dims,
                           quantization=quant),
                TensorSpec("input_b", dtype, shape, shape_signature=signature_dims,
                           quantization=quant),
                TensorSpec("output", dtype, shape, shape_signature=signature_dims,
                           quantization=quant),
            ),
            ops=(OpSpec(sig.op, (0, 1), (2,), version=version, builtin_options=options),),
            inputs=(0, 1),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if sig.op == "RESHAPE":
        if dynamic:
            raise UnprobeableSignatureError(
                "dynamic RESHAPE has no template (static shapes only)"
            )
        quant = _INT8_ACT_QPARAMS if dtype == "int8" else None
        flat = (_elements(shape),)
        shape_data = struct.pack(f"<{rank}i", *shape)
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, flat, quantization=quant),
                TensorSpec("shape", "int32", (rank,), data=shape_data),
                TensorSpec("output", dtype, shape, quantization=quant),
            ),
            ops=(OpSpec("RESHAPE", (0, 1), (2,)),),
            inputs=(0,),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if sig.op in _WEIGHTED_OPS:
        if dynamic:
            raise UnprobeableSignatureError(
                f"dynamic-shape {sig.op} has no template (static shapes only)"
            )
        return _weighted_graph(sig.op, dtype, rank, shape)

    if sig.op == "DEQUANTIZE":
        if dtype != "float32":
            raise UnprobeableSignatureError(
                f"DEQUANTIZE probe template dequantizes float16 to float32 "
                f"(got output dtype {dtype!r})"
            )
        # Representative input form: float16 — the dominant zoo form (fp16
        # post-training quantization stores fp16 weights behind DEQUANTIZE).
        # int8-weight DEQUANTIZE shares the output signature; the fixture
        # itself documents which form was measured (micrograph honesty rule).
        return GraphSpec(
            tensors=(
                TensorSpec("input", "float16", shape, shape_signature=signature_dims),
                TensorSpec("output", dtype, shape, shape_signature=signature_dims),
            ),
            ops=(OpSpec("DEQUANTIZE", (0,), (1,), version=3),),
            inputs=(0,),
            outputs=(1,),
            description=_PROBE_DESCRIPTION,
        )

    raise UnprobeableSignatureError(f"no probe template for op {sig.op}")


def _weighted_graph(op: str, dtype: str, rank: int, shape: tuple[int, ...]) -> GraphSpec:
    """Weighted/structured fp32-activation templates (static shapes only).

    `shape` is the OUTPUT shape for the requested rank; constants (weights,
    bias, perm/begin/size/axes/paddings/size vectors) are deterministic, keyed
    by (op, dtype, rank). Quantized-weight forms have no template here — they
    stay unprobeable with a precise reason.
    """
    key = f"{op}:{dtype}:{rank}"
    depth = _PROBE_DEPTH
    int8 = dtype == "int8"

    # CAST and the comparisons carry their own dtype semantics; SELECT is
    # feedable for float32/int32; the _INT8_WEIGHTED families additionally
    # accept int8 activations with the canonical qparams. Everything else is
    # fp32-activation only.
    if op in _COMPARISONS or op == "CAST":
        pass
    elif op in _SELECTS:
        if dtype not in ("float32", "int32"):
            raise UnprobeableSignatureError(
                f"{op} probe template supports float32/int32 outputs (got {dtype!r})"
            )
    elif int8 and op in _INT8_WEIGHTED:
        pass
    elif dtype != "float32":
        raise UnprobeableSignatureError(
            f"{op} probe template supports fp32 activations"
            + (" and int8" if op in _INT8_WEIGHTED else "")
            + f" (got {dtype!r})"
        )

    if op in _RANK4_ONLY and rank != 4:
        raise UnprobeableSignatureError(f"{op} probe requires rank 4 (NHWC)")

    if op in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        batch, height, width, out_ch = shape
        depthwise = op == "DEPTHWISE_CONV_2D"
        in_ch = out_ch if depthwise else depth
        filter_shape = (1, 2, 2, out_ch) if depthwise else (out_ch, 2, 2, in_ch)
        options = _DEPTHWISE_CONV_2D_OPTIONS if depthwise else _CONV_2D_OPTIONS
        if int8:
            # Per-channel filter quantization (the zoo's channelwise form);
            # int8 conv kernels require symmetric filters (zero_point 0) and
            # bias_scale == input_scale * filter_scale per channel.
            filter_tensor = TensorSpec(
                "filter", dtype, filter_shape,
                data=_const_int8s(_elements(filter_shape), f"{key}:filter"),
                quantization=_per_channel_qparams(
                    out_ch, 3 if depthwise else 0, _INT8_WEIGHT_SCALE
                ),
            )
            bias_tensor = TensorSpec(
                "bias", "int32", (out_ch,), data=_const_int32s(out_ch, f"{key}:bias"),
                quantization=_per_channel_qparams(out_ch, 0, _INT8_BIAS_SCALE),
            )
        else:
            filter_tensor = TensorSpec(
                "filter", dtype, filter_shape,
                data=_const_floats(_elements(filter_shape), f"{key}:filter"),
            )
            bias_tensor = TensorSpec(
                "bias", dtype, (out_ch,), data=_const_floats(out_ch, f"{key}:bias")
            )
        act_quant = _INT8_ACT_QPARAMS if int8 else None
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, (batch, height, width, in_ch),
                           quantization=act_quant),
                filter_tensor,
                bias_tensor,
                TensorSpec("output", dtype, shape, quantization=act_quant),
            ),
            # int8 per-channel convolutions are op version 3.
            ops=(OpSpec(op, (0, 1, 2), (3,), version=3 if int8 else 1,
                        builtin_options=options),),
            inputs=(0,),
            outputs=(3,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "TRANSPOSE_CONV":
        batch, height, width, out_ch = shape
        filter_shape = (out_ch, 2, 2, depth)
        return GraphSpec(
            tensors=(
                TensorSpec("output_shape", "int32", (4,), data=_int32_data(shape)),
                TensorSpec(
                    "filter", dtype, filter_shape,
                    data=_const_floats(_elements(filter_shape), f"{key}:filter"),
                ),
                TensorSpec("input", dtype, (batch, height, width, depth)),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1, 2), (3,), builtin_options=_TRANSPOSE_CONV_OPTIONS),),
            inputs=(2,),
            outputs=(3,),
            description=_PROBE_DESCRIPTION,
        )

    if op in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
        act_quant = _INT8_ACT_QPARAMS if int8 else None
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, shape, quantization=act_quant),
                TensorSpec("output", dtype, shape, quantization=act_quant),
            ),
            # int8 pools are op version 2.
            ops=(OpSpec(op, (0,), (1,), version=2 if int8 else 1,
                        builtin_options=_POOL_2D_OPTIONS),),
            inputs=(0,),
            outputs=(1,),
            description=_PROBE_DESCRIPTION,
        )

    if op in _RESIZES:
        batch, height, width, channels = shape
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, (batch, 2, 2, channels)),
                TensorSpec("size", "int32", (2,), data=_int32_data((height, width))),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1), (2,), builtin_options=_RESIZES[op]),),
            inputs=(0,),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "FULLY_CONNECTED":
        if rank < 2:
            raise UnprobeableSignatureError("FULLY_CONNECTED probe requires rank >= 2")
        if int8 and rank != 2:
            raise UnprobeableSignatureError(
                "int8 FULLY_CONNECTED probe requires rank 2 (keep_num_dims "
                "forms stay fp32-only)"
            )
        units = shape[-1]
        input_shape = (*shape[:-1], depth)
        keep_dims = rank > 2  # keep_num_dims preserves batch dims → op version 5
        if int8:
            # Per-tensor symmetric weights — the int8 FC form (op version 4).
            act_quant = _INT8_ACT_QPARAMS
            weights_tensor = TensorSpec(
                "weights", dtype, (units, depth),
                data=_const_int8s(units * depth, f"{key}:weights"),
                quantization=QuantizationSpec(
                    scale=(_INT8_WEIGHT_SCALE,), zero_point=(0,)
                ),
            )
            bias_tensor = TensorSpec(
                "bias", "int32", (units,), data=_const_int32s(units, f"{key}:bias"),
                quantization=QuantizationSpec(scale=(_INT8_BIAS_SCALE,), zero_point=(0,)),
            )
            version = 4
        else:
            act_quant = None
            weights_tensor = TensorSpec(
                "weights", dtype, (units, depth),
                data=_const_floats(units * depth, f"{key}:weights"),
            )
            bias_tensor = TensorSpec(
                "bias", dtype, (units,), data=_const_floats(units, f"{key}:bias")
            )
            version = 5 if keep_dims else 1
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, input_shape, quantization=act_quant),
                weights_tensor,
                bias_tensor,
                TensorSpec("output", dtype, shape, quantization=act_quant),
            ),
            ops=(
                OpSpec(
                    op, (0, 1, 2), (3,),
                    version=version,
                    builtin_options=_FC_KEEP_DIMS_OPTIONS if keep_dims else _FC_OPTIONS,
                ),
            ),
            inputs=(0,),
            outputs=(3,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "BATCH_MATMUL":
        if rank < 2:
            raise UnprobeableSignatureError("BATCH_MATMUL probe requires rank >= 2")
        rows, cols = shape[-2], shape[-1]
        return GraphSpec(
            tensors=(
                TensorSpec("input_a", dtype, (*shape[:-2], rows, depth)),
                TensorSpec("input_b", dtype, (*shape[:-2], depth, cols)),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1), (2,), builtin_options=_BATCH_MATMUL_OPTIONS),),
            inputs=(0, 1),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "CONCATENATION":
        if rank < 1:
            raise UnprobeableSignatureError("CONCATENATION probe requires rank >= 1")
        half = (*shape[:-1], shape[-1] // 2)
        options = BuiltinOptionsSpec(
            type_code=_CONCATENATION_TYPE,
            fields=(OptionsField(slot=0, kind="int32", value=rank - 1),),
        )
        return GraphSpec(
            tensors=(
                TensorSpec("input_a", dtype, half),
                TensorSpec("input_b", dtype, half),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1), (2,), builtin_options=options),),
            inputs=(0, 1),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "TRANSPOSE":
        if rank < 1:
            raise UnprobeableSignatureError("TRANSPOSE probe requires rank >= 1")
        if int8 and rank >= 5:
            raise UnprobeableSignatureError(
                "int8 TRANSPOSE probe supports rank <= 4"
            )
        act_quant = _INT8_ACT_QPARAMS if int8 else None
        perm = tuple(range(rank - 1, -1, -1))
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, tuple(reversed(shape)), quantization=act_quant),
                TensorSpec("perm", "int32", (rank,), data=_int32_data(perm)),
                TensorSpec("output", dtype, shape, quantization=act_quant),
            ),
            # 5-D+ TRANSPOSE is op version 4; int8 is version 2.
            ops=(
                OpSpec(
                    op, (0, 1), (2,),
                    version=4 if rank >= 5 else (2 if int8 else 1),
                    builtin_options=_TRANSPOSE_OPTIONS,
                ),
            ),
            inputs=(0,),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "SLICE":
        if rank < 1:
            raise UnprobeableSignatureError("SLICE probe requires rank >= 1")
        input_shape = tuple(dim + 1 for dim in shape)
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, input_shape),
                TensorSpec("begin", "int32", (rank,), data=_int32_data((1,) * rank)),
                TensorSpec("size", "int32", (rank,), data=_int32_data(shape)),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1, 2), (3,), builtin_options=_SLICE_OPTIONS),),
            inputs=(0,),
            outputs=(3,),
            description=_PROBE_DESCRIPTION,
        )

    if op in _REDUCERS:
        if rank < 1:
            raise UnprobeableSignatureError(f"{op} probe requires rank >= 1")
        input_shape = shape
        output_shape = (1, *shape[1:])  # axes=[0], keep_dims=true
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, input_shape),
                TensorSpec("axes", "int32", (1,), data=_int32_data((0,))),
                TensorSpec("output", dtype, output_shape),
            ),
            ops=(OpSpec(op, (0, 1), (2,), builtin_options=_REDUCER_OPTIONS),),
            inputs=(0,),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "PAD":
        if rank < 1:
            raise UnprobeableSignatureError("PAD probe requires rank >= 1")
        if shape[-1] < 3:
            raise UnprobeableSignatureError("PAD probe requires last dim >= 3")
        act_quant = _INT8_ACT_QPARAMS if int8 else None
        input_shape = (*shape[:-1], shape[-1] - 2)
        paddings = (*(0 for _ in range(2 * (rank - 1))), 1, 1)
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, input_shape, quantization=act_quant),
                TensorSpec("paddings", "int32", (rank, 2), data=_int32_data(paddings)),
                TensorSpec("output", dtype, shape, quantization=act_quant),
            ),
            # int8 PAD is op version 2.
            ops=(OpSpec(op, (0, 1), (2,), version=2 if int8 else 1,
                        builtin_options=_PAD_OPTIONS),),
            inputs=(0,),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "LEAKY_RELU":
        return GraphSpec(
            tensors=(
                TensorSpec("input", dtype, shape),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0,), (1,), builtin_options=_LEAKY_RELU_OPTIONS),),
            inputs=(0,),
            outputs=(1,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "CAST":
        # The signature names the OUTPUT dtype (classification contract); the
        # input dtype is the representative counterpart: int32 for float-family
        # outputs, float32 for integer/bool outputs.
        input_dtype = "int32" if dtype.startswith(("float", "bfloat")) else "float32"
        return GraphSpec(
            tensors=(
                TensorSpec("input", input_dtype, shape),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0,), (1,), builtin_options=_CAST_OPTIONS),),
            inputs=(0,),
            outputs=(1,),
            description=_PROBE_DESCRIPTION,
        )

    if op in _COMPARISONS:
        if dtype != "bool":
            raise UnprobeableSignatureError(
                f"{op} output dtype is bool; signature carries {dtype!r}"
            )
        return GraphSpec(
            tensors=(
                TensorSpec("input_a", "float32", shape),
                TensorSpec("input_b", "float32", shape),
                TensorSpec("output", "bool", shape),
            ),
            ops=(OpSpec(op, (0, 1), (2,), builtin_options=_COMPARISONS[op]),),
            inputs=(0, 1),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    if op in _SELECTS:
        return GraphSpec(
            tensors=(
                TensorSpec("condition", "bool", shape),
                TensorSpec("input_a", dtype, shape),
                TensorSpec("input_b", dtype, shape),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1, 2), (3,), builtin_options=_SELECTS[op]),),
            inputs=(0, 1, 2),
            outputs=(3,),
            description=_PROBE_DESCRIPTION,
        )

    if op == "EMBEDDING_LOOKUP":
        if rank < 2:
            raise UnprobeableSignatureError("EMBEDDING_LOOKUP probe requires rank >= 2")
        # 8 table rows: the seeded-feeds contract draws integer inputs from
        # 1..4, so every lookup id is in range by construction.
        table_shape = (8, *shape[1:])
        return GraphSpec(
            tensors=(
                TensorSpec("lookup", "int32", (shape[0],)),
                TensorSpec(
                    "table", dtype, table_shape,
                    data=_const_floats(_elements(table_shape), f"{key}:table"),
                ),
                TensorSpec("output", dtype, shape),
            ),
            ops=(OpSpec(op, (0, 1), (2,)),),
            inputs=(0,),
            outputs=(2,),
            description=_PROBE_DESCRIPTION,
        )

    raise UnprobeableSignatureError(f"no probe template for op {op}")


def _verify_round_trip(sig: ProbeSignature, data: bytes) -> None:
    """The generated fixture must classify back to exactly the requested
    signature — otherwise the probe would measure something else."""
    parsed = parse_tflite(data)
    subgraph = parsed.subgraphs[0]
    if len(subgraph.nodes) != 1:
        raise RuntimeError(f"probe fixture for {sig.op} is not single-op")
    dtypes, shape_meta = node_signature(subgraph, subgraph.nodes[0])
    if tuple(dtypes) != sig.dtypes or shape_meta != sig.shape_meta:
        raise RuntimeError(
            f"probe fixture round-trip mismatch for {sig.op}: "
            f"built ({dtypes}, {shape_meta}), requested "
            f"({list(sig.dtypes)}, {sig.shape_meta})"
        )


def generate_fixture(sig: ProbeSignature, out_dir: Path) -> ProbeFixture:
    """Generate (or re-generate, byte-identically) the fixture for a signature."""
    graph = build_probe_graph(sig)
    data = build_tflite(graph)
    _verify_round_trip(sig, data)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sig.fixture_id}.tflite"
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    return ProbeFixture(signature=sig, fixture_id=sig.fixture_id, path=path)
