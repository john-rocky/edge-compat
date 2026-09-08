"""Builtin-options registries for faithful round-trips and rule-emitted nodes.

Two closed registries, both extended additively (the probe-template pattern):

- `OPTION_LAYOUTS`: BuiltinOptions union type code → the scalar fields the
  loader knows how to read back byte-faithfully. A populated field outside the
  layout (or an unregistered type code) makes the model unsupported for
  rewriting — refused with a precise reason, never silently dropped.
- `default_options_for`: op name → the builtin options a rule-emitted node
  gets (the empty options table converters emit, of the right union type;
  SOFTMAX gets beta=1.0). Emittable ops are exactly those whose canonical
  table is complete and valid on its own — per-instance options (conv/pool
  strides, CONCATENATION axis) cannot be defaulted, so those ops cannot be
  targets of replace_op/decompose in v1.

Union type codes mirror the BuiltinOptions union in schema.fbs (append-only).
Codes were cross-checked against real converter output (litert-community zoo,
2026-08-12): the census anchors PACK=59 and RESIZE_NEAREST_NEIGHBOR=74 /
LEAKY_RELU=75 pin the union run in between, which corrected three earlier
entries (FloorMod 69→72, SquaredDifference 73→76, Abs 75→78).
"""

from __future__ import annotations

from dataclasses import dataclass

from litert_compat.parser.builder import BuiltinOptionsSpec, OptionsField
from litert_compat.probe.gen import BINARY_ELEMENTWISE, UNARY_ELEMENTWISE

CAST_OPTIONS_TYPE = 37  # CastOptions
_SOFTMAX_OPTIONS_TYPE = 9  # SoftmaxOptions
_RESHAPE_OPTIONS_TYPE = 17  # ReshapeOptions
_NEG_OPTIONS_TYPE = 42  # NegOptions
_EXP_OPTIONS_TYPE = 33  # ExpOptions
_SQUARE_OPTIONS_TYPE = 66  # SquareOptions
_ABS_OPTIONS_TYPE = 78  # AbsOptions


@dataclass(frozen=True)
class OptionsFieldLayout:
    """One scalar field of a known builtin-options table."""

    name: str
    slot: int
    kind: str  # "float32" | "int32" | "int8" | "uint8" | "bool"


# Scalar fields the loader can read back faithfully, per union type code.
# An empty tuple means the table is known and carries no readable scalars —
# only a fully-default (empty) table round-trips; any populated slot refuses.
OPTION_LAYOUTS: dict[int, tuple[OptionsFieldLayout, ...]] = {
    1: (  # Conv2DOptions
        OptionsFieldLayout("padding", 0, "int8"),
        OptionsFieldLayout("stride_w", 1, "int32"),
        OptionsFieldLayout("stride_h", 2, "int32"),
        OptionsFieldLayout("fused_activation_function", 3, "int8"),
        OptionsFieldLayout("dilation_w_factor", 4, "int32"),
        OptionsFieldLayout("dilation_h_factor", 5, "int32"),
        OptionsFieldLayout("quantized_bias_type", 6, "int8"),
    ),
    2: (  # DepthwiseConv2DOptions
        OptionsFieldLayout("padding", 0, "int8"),
        OptionsFieldLayout("stride_w", 1, "int32"),
        OptionsFieldLayout("stride_h", 2, "int32"),
        OptionsFieldLayout("depth_multiplier", 3, "int32"),
        OptionsFieldLayout("fused_activation_function", 4, "int8"),
        OptionsFieldLayout("dilation_w_factor", 5, "int32"),
        OptionsFieldLayout("dilation_h_factor", 6, "int32"),
    ),
    5: (  # Pool2DOptions
        OptionsFieldLayout("padding", 0, "int8"),
        OptionsFieldLayout("stride_w", 1, "int32"),
        OptionsFieldLayout("stride_h", 2, "int32"),
        OptionsFieldLayout("filter_width", 3, "int32"),
        OptionsFieldLayout("filter_height", 4, "int32"),
        OptionsFieldLayout("fused_activation_function", 5, "int8"),
    ),
    8: (  # FullyConnectedOptions
        OptionsFieldLayout("fused_activation_function", 0, "int8"),
        OptionsFieldLayout("weights_format", 1, "int8"),
        OptionsFieldLayout("keep_num_dims", 2, "bool"),
        OptionsFieldLayout("asymmetric_quantize_inputs", 3, "bool"),
        OptionsFieldLayout("quantized_bias_type", 4, "int8"),
    ),
    _SOFTMAX_OPTIONS_TYPE: (OptionsFieldLayout("beta", 0, "float32"),),
    10: (  # ConcatenationOptions
        OptionsFieldLayout("axis", 0, "int32"),
        OptionsFieldLayout("fused_activation_function", 1, "int8"),
    ),
    11: (  # AddOptions
        OptionsFieldLayout("fused_activation_function", 0, "int8"),
        OptionsFieldLayout("pot_scale_int16", 1, "bool"),
    ),
    15: (  # ResizeBilinearOptions; slots 0/1 (deprecated new_height/new_width) refuse
        OptionsFieldLayout("align_corners", 2, "bool"),
        OptionsFieldLayout("half_pixel_centers", 3, "bool"),
    ),
    _RESHAPE_OPTIONS_TYPE: (),  # new_shape is a vector: populated → refuse
    21: (OptionsFieldLayout("fused_activation_function", 0, "int8"),),  # MulOptions
    22: (),  # PadOptions
    26: (),  # TransposeOptions
    27: (OptionsFieldLayout("keep_dims", 0, "bool"),),  # ReducerOptions
    28: (  # SubOptions
        OptionsFieldLayout("fused_activation_function", 0, "int8"),
        OptionsFieldLayout("pot_scale_int16", 1, "bool"),
    ),
    29: (OptionsFieldLayout("fused_activation_function", 0, "int8"),),  # DivOptions
    32: (  # StridedSliceOptions
        OptionsFieldLayout("begin_mask", 0, "int32"),
        OptionsFieldLayout("end_mask", 1, "int32"),
        OptionsFieldLayout("ellipsis_mask", 2, "int32"),
        OptionsFieldLayout("new_axis_mask", 3, "int32"),
        OptionsFieldLayout("shrink_axis_mask", 4, "int32"),
        OptionsFieldLayout("offset", 5, "bool"),
    ),
    _EXP_OPTIONS_TYPE: (),  # ExpOptions
    CAST_OPTIONS_TYPE: (  # CastOptions: TensorType bytes, resolved from tensors at runtime
        OptionsFieldLayout("in_data_type", 0, "int8"),
        OptionsFieldLayout("out_data_type", 1, "int8"),
    ),
    39: (),  # MaximumMinimumOptions
    40: (OptionsFieldLayout("output_type", 0, "int8"),),  # ArgMaxOptions
    41: (),  # LessOptions
    _NEG_OPTIONS_TYPE: (),  # NegOptions
    43: (),  # PadV2Options
    44: (),  # GreaterOptions
    45: (),  # GreaterEqualOptions
    46: (),  # LessEqualOptions
    47: (),  # SelectOptions
    48: (),  # SliceOptions
    53: (),  # EqualOptions
    54: (),  # NotEqualOptions
    49: (  # TransposeConvOptions
        OptionsFieldLayout("padding", 0, "int8"),
        OptionsFieldLayout("stride_w", 1, "int32"),
        OptionsFieldLayout("stride_h", 2, "int32"),
        OptionsFieldLayout("fused_activation_function", 3, "int8"),
        OptionsFieldLayout("quantized_bias_type", 4, "int8"),
    ),
    56: (),  # PowOptions
    59: (  # PackOptions
        OptionsFieldLayout("values_count", 0, "int32"),
        OptionsFieldLayout("axis", 1, "int32"),
    ),
    65: (),  # FloorDivOptions
    _SQUARE_OPTIONS_TYPE: (),  # SquareOptions
    72: (),  # FloorModOptions
    74: (  # ResizeNearestNeighborOptions
        OptionsFieldLayout("align_corners", 0, "bool"),
        OptionsFieldLayout("half_pixel_centers", 1, "bool"),
    ),
    75: (OptionsFieldLayout("alpha", 0, "float32"),),  # LeakyReluOptions
    76: (),  # SquaredDifferenceOptions
    _ABS_OPTIONS_TYPE: (),  # AbsOptions
    91: (),  # HardSwishOptions
    98: (),  # SelectV2Options
    101: (  # BatchMatMulOptions
        OptionsFieldLayout("adj_x", 0, "bool"),
        OptionsFieldLayout("adj_y", 1, "bool"),
        OptionsFieldLayout("asymmetric_quantize_inputs", 2, "bool"),
    ),
    102: (  # CumsumOptions
        OptionsFieldLayout("exclusive", 0, "bool"),
        OptionsFieldLayout("reverse", 1, "bool"),
    ),
    116: (OptionsFieldLayout("approximate", 0, "bool"),),  # GeluOptions
}


class UnknownOptionsOpError(ValueError):
    """The op is outside the emission registry — a rule cannot emit it in v1."""


# Ops a rule may emit because the EMPTY options table of the right union type
# is a complete, valid semantic (every field at its schema default carries a
# well-defined meaning). Ops whose options are per-instance and load-bearing —
# convolution/pool strides, CONCATENATION's axis — cannot be canonically
# defaulted and stay outside: extending them needs per-node option params in
# the rule schema, not a registry entry. (Extended 2026-08-12 from the
# original CAST/RESHAPE/binary set — DECISIONS #158.)
_EMPTY_TABLE_EMISSION: dict[str, int] = {
    "PAD": 22,
    "PADV2": 43,
    "TRANSPOSE": 26,
    "SLICE": 48,
    "SELECT": 47,
    "SELECT_V2": 98,
    "GREATER": 44,
    "GREATER_EQUAL": 45,
    "LESS": 41,
    "LESS_EQUAL": 46,
    "EQUAL": 53,
    "NOT_EQUAL": 54,
    "HARD_SWISH": 91,
    "GELU": 116,  # approximate=false: the exact erf form
    "RESIZE_BILINEAR": 15,  # align_corners/half_pixel_centers false
    "RESIZE_NEAREST_NEIGHBOR": 74,
    "FULLY_CONNECTED": 8,  # activation NONE, keep_num_dims false
    "BATCH_MATMUL": 101,  # adj_x/adj_y false
    "SUM": 27,  # keep_dims false
    "MEAN": 27,
    "REDUCE_MAX": 27,
    "REDUCE_MIN": 27,
}

# SOFTMAX is the one emission whose canonical table is NOT empty: beta=1.0 is
# the softmax (beta 0 would be semantically broken), matching the probe
# template and what converters emit.
_SOFTMAX_EMISSION = BuiltinOptionsSpec(
    type_code=_SOFTMAX_OPTIONS_TYPE,
    fields=(OptionsField(slot=0, kind="float32", value=1.0),),
)


def default_options_for(op: str) -> BuiltinOptionsSpec | None:
    """The builtin options a rule-emitted node of `op` carries: none for the
    option-free unary set, the empty options table of the right union type for
    registered binary elementwise ops, CAST, and the `_EMPTY_TABLE_EMISSION`
    families, and beta=1.0 for SOFTMAX. Raises for anything else — including
    ops whose options are per-instance and load-bearing (convolution/pool
    strides, CONCATENATION's axis), which no registry default can express."""
    if op in UNARY_ELEMENTWISE or op == "RESHAPE":
        return None
    if op in BINARY_ELEMENTWISE:
        return BuiltinOptionsSpec(type_code=BINARY_ELEMENTWISE[op])
    if op == "CAST":
        return BuiltinOptionsSpec(type_code=CAST_OPTIONS_TYPE)
    if op == "SOFTMAX":
        return _SOFTMAX_EMISSION
    if op in _EMPTY_TABLE_EMISSION:
        return BuiltinOptionsSpec(type_code=_EMPTY_TABLE_EMISSION[op])
    raise UnknownOptionsOpError(
        f"op {op} is outside the v1 options registry (option-free unary ops, "
        "binary elementwise ops, CAST, RESHAPE, SOFTMAX, and the empty-table "
        "families: pads, TRANSPOSE, SLICE, selects, comparisons, resizes, "
        "reducers, FULLY_CONNECTED, BATCH_MATMUL, HARD_SWISH, GELU) — "
        "extending the registry is additive, but a rule cannot emit an op "
        "whose options contract is unknown or per-instance (conv/pool "
        "strides, CONCATENATION axis)"
    )
