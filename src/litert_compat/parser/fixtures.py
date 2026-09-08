"""Synthetic `.tflite` example fixtures, designed against the example matrix.

The committed fixtures in `data/examples/` are the byte output of this module;
`python -m litert_compat.parser.fixtures [out_dir]` regenerates them and tests
assert byte-identity. Graphs are synthetic (weightless, dtype jumps allowed) —
they exercise the static linter, which never executes a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    OpSpec,
    TensorSpec,
    build_tflite,
)


def fixture_clean() -> bytes:
    """Fully delegated against the example matrix: 1 partition, 0 blocking ops."""
    tensors = (
        TensorSpec("input", "float32", (1, 8, 8, 3)),
        TensorSpec("conv1_filter", "float32", (8, 3, 3, 3)),
        TensorSpec("conv1_bias", "float32", (8,)),
        TensorSpec("conv1_out", "float32", (1, 8, 8, 8)),
        TensorSpec("pool_out", "float32", (1, 4, 4, 8)),
        TensorSpec("conv2_filter", "float32", (8, 1, 1, 8)),
        TensorSpec("conv2_bias", "float32", (8,)),
        TensorSpec("output", "float32", (1, 4, 4, 8)),
    )
    ops = (
        OpSpec("CONV_2D", (0, 1, 2), (3,)),
        OpSpec("AVERAGE_POOL_2D", (3,), (4,)),
        OpSpec("CONV_2D", (4, 5, 6), (7,)),
    )
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(7,)))


def fixture_split() -> bytes:
    """Two GPU partitions split by a fallback GATHER_ND (with rewrite hint),
    an unknown RESHAPE, and an unknown-dtype SOFTMAX (float32 vs the matrix's
    float16 entry) — exercises partitions, hints, and the needs-probe list."""
    tensors = (
        TensorSpec("input", "float32", (1, 16, 16, 3)),
        TensorSpec("conv_filter", "float32", (8, 3, 3, 3)),
        TensorSpec("conv_bias", "float32", (8,)),
        TensorSpec("conv_out", "float32", (1, 16, 16, 8)),
        TensorSpec("gather_indices", "int32", (128, 3)),
        TensorSpec("gather_out", "float32", (128, 8)),
        TensorSpec("reshape_shape", "int32", (2,)),
        TensorSpec("reshape_out", "float32", (1, 1024)),
        TensorSpec("fc_weights", "float32", (10, 1024)),
        TensorSpec("fc_bias", "float32", (10,)),
        TensorSpec("fc_out", "float32", (1, 10)),
        TensorSpec("softmax_out", "float32", (1, 10)),
    )
    ops = (
        OpSpec("CONV_2D", (0, 1, 2), (3,)),
        OpSpec("GATHER_ND", (3, 4), (5,)),
        OpSpec("RESHAPE", (5, 6), (7,)),
        OpSpec("FULLY_CONNECTED", (7, 8, 9), (10,)),
        OpSpec("SOFTMAX", (10,), (11,)),
    )
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(11,)))


def fixture_web_a() -> bytes:
    """Web-runnable: option-free elementwise ops, no constant tensors, so the
    real LiteRT runtime (including LiteRT.js) can execute it — unlike the
    weightless graphs above, which only exercise the static linter."""
    tensors = (
        TensorSpec("input", "float32", (1, 64)),
        TensorSpec("neg_out", "float32", (1, 64)),
        TensorSpec("output", "float32", (1, 64)),
    )
    ops = (
        OpSpec("NEG", (0,), (1,)),
        OpSpec("LOGISTIC", (1,), (2,)),
    )
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(2,)))


def fixture_web_b() -> bytes:
    """Web-runnable (see fixture_web_a): a 4D elementwise chain."""
    tensors = (
        TensorSpec("input", "float32", (1, 8, 8, 3)),
        TensorSpec("abs_out", "float32", (1, 8, 8, 3)),
        TensorSpec("sqrt_out", "float32", (1, 8, 8, 3)),
        TensorSpec("output", "float32", (1, 8, 8, 3)),
    )
    ops = (
        OpSpec("ABS", (0,), (1,)),
        OpSpec("SQRT", (1,), (2,)),
        OpSpec("TANH", (2,), (3,)),
    )
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(3,)))


# Deliberately NOT a flatbuffer: the broken entry in the example web catalog
# (crash-isolation path of the Phase 5 sweep). Deterministic bytes.
# The marker keeps the pre-rename spelling on purpose: the fixture bytes are
# pinned by tests and by the sha256 in the committed example cards and goldens.
BROKEN_FIXTURE = b"litert-compat deliberately broken example model\x00" * 4


# --- Phase 10 fix-demo fixtures ---------------------------------------------
# Runnable single-op graphs (real buffers/options where the runtime needs
# them), each paired with one example transform rule in
# data/examples/transforms/ and the statuses in matrix_fix_example.json.

_DIV_OPTIONS = BuiltinOptionsSpec(type_code=29)  # empty DivOptions
_MUL_OPTIONS = BuiltinOptionsSpec(type_code=21)  # empty MulOptions


def fixture_fix_div() -> bytes:
    """int32 DIV — fallback in the fix-demo matrix; the replace_op example
    (`example-replace-div-floor-div`) rewrites it to FLOOR_DIV."""
    tensors = (
        TensorSpec("input_a", "int32", (1, 16)),
        TensorSpec("input_b", "int32", (1, 16)),
        TensorSpec("output", "int32", (1, 16)),
    )
    ops = (OpSpec("DIV", (0, 1), (2,), builtin_options=_DIV_OPTIONS),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0, 1), outputs=(2,)))


def fixture_fix_square() -> bytes:
    """float32 SQUARE — fallback in the fix-demo matrix; the decompose example
    (`example-decompose-square-mul`) rewrites it to MUL(x, x)."""
    tensors = (
        TensorSpec("input", "float32", (1, 4, 8)),
        TensorSpec("output", "float32", (1, 4, 8)),
    )
    ops = (OpSpec("SQUARE", (0,), (1,)),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(1,)))


def fixture_fix_int_mul() -> bytes:
    """int32 MUL — fallback in the fix-demo matrix; the insert_cast example
    (`example-insert-cast-mul-int32`) computes it in float32."""
    tensors = (
        TensorSpec("input_a", "int32", (1, 16)),
        TensorSpec("input_b", "int32", (1, 16)),
        TensorSpec("output", "int32", (1, 16)),
    )
    ops = (OpSpec("MUL", (0, 1), (2,), builtin_options=_MUL_OPTIONS),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0, 1), outputs=(2,)))


def fixture_fix_io64() -> bytes:
    """int64 I/O around a delegated NEG — the io_cast example
    (`example-io-cast-int64`) re-exposes the interface as int32."""
    tensors = (
        TensorSpec("input", "int64", (1, 16)),
        TensorSpec("output", "int64", (1, 16)),
    )
    ops = (OpSpec("NEG", (0,), (1,)),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(1,)))


def fixture_mixed() -> bytes:
    """Dynamic-batch int8 FULLY_CONNECTED (fallback), a custom op (unknown),
    a delegated CONV_2D, a float16 SOFTMAX (incorrect — delegated but wrong
    numerics), and a TRANSPOSE_CONV (fallback)."""
    tensors = (
        TensorSpec("input", "int8", (1, 64), shape_signature=(-1, 64)),
        TensorSpec("fc_weights", "int8", (32, 64)),
        TensorSpec("fc_bias", "int32", (32,)),
        TensorSpec("fc_out", "int8", (1, 32), shape_signature=(-1, 32)),
        TensorSpec("custom_out", "float32", (1, 4, 4, 2)),
        TensorSpec("conv_filter", "float32", (4, 1, 1, 2)),
        TensorSpec("conv_bias", "float32", (4,)),
        TensorSpec("conv_out", "float32", (1, 4, 4, 4)),
        TensorSpec("softmax_out", "float16", (1, 4, 4, 4)),
        TensorSpec("deconv_output_shape", "int32", (4,)),
        TensorSpec("deconv_filter", "float32", (2, 2, 2, 4)),
        TensorSpec("output", "float32", (1, 8, 8, 2)),
    )
    ops = (
        OpSpec("FULLY_CONNECTED", (0, 1, 2), (3,)),
        OpSpec("CUSTOM", (3,), (4,), custom_code="ExampleCustomOp"),
        OpSpec("CONV_2D", (4, 5, 6), (7,)),
        OpSpec("SOFTMAX", (7,), (8,)),
        OpSpec("TRANSPOSE_CONV", (9, 10, 8), (11,)),
    )
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(11,)))


FIXTURES: dict[str, bytes] = {
    "model_clean_example.tflite": fixture_clean(),
    "model_split_example.tflite": fixture_split(),
    "model_mixed_example.tflite": fixture_mixed(),
    "model_web_a_example.tflite": fixture_web_a(),
    "model_web_b_example.tflite": fixture_web_b(),
    "model_broken_example.tflite": BROKEN_FIXTURE,
    "model_fix_div_example.tflite": fixture_fix_div(),
    "model_fix_square_example.tflite": fixture_fix_square(),
    "model_fix_int_mul_example.tflite": fixture_fix_int_mul(),
    "model_fix_io64_example.tflite": fixture_fix_io64(),
}


def write_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in FIXTURES.items():
        path = out_dir / name
        path.write_bytes(data)
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/examples")
    for written_path in write_all(target):
        print(written_path)
