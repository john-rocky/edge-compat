"""Synthetic fixtures for the STAGED (draft, owner-review-pending) transform
rules in data/transforms_staging/.

Same machinery as data/examples/ (litert_compat.parser.fixtures /
parser.builder): minimal runnable single-op graphs exhibiting each rule's
symptom. Regenerate with:

    uv run python data/transforms_staging/fixtures/make_fixtures.py

Deterministic: same specs -> same bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path

from litert_compat.parser.builder import (
    BuiltinOptionsSpec, GraphSpec, OpSpec, TensorSpec, build_tflite,
)


def fixture_select() -> bytes:
    """float32 SELECT on a bool mask — the ML Drift GPU SELECT symptom
    (gpu-clean-conversion SKILL.md symptom table). Rewritten by the
    select-to-arithmetic draft into CAST+SUB+MUL+ADD."""
    tensors = (
        TensorSpec("cond", "bool", (1, 16)),
        TensorSpec("then_value", "float32", (1, 16)),
        TensorSpec("else_value", "float32", (1, 16)),
        TensorSpec("output", "float32", (1, 16)),
    )
    ops = (OpSpec("SELECT", (0, 1, 2), (3,)),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0, 1, 2), outputs=(3,)))


def fixture_select_v2() -> bytes:
    """Same symptom as fixture_select for the SELECT_V2 builtin (the skill
    names both). Rewritten by select-v2-to-arithmetic."""
    tensors = (
        TensorSpec("cond", "bool", (1, 16)),
        TensorSpec("then_value", "float32", (1, 16)),
        TensorSpec("else_value", "float32", (1, 16)),
        TensorSpec("output", "float32", (1, 16)),
    )
    ops = (OpSpec("SELECT_V2", (0, 1, 2), (3,)),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0, 1, 2), outputs=(3,)))


def fixture_int64_input() -> bytes:
    """int64 graph input feeding an interior int64->float32 CAST + ABS — the
    'FLOAT input, not int' interface symptom (LITERT_CONVERSION_GUIDE.md,
    DeepPhonemizer section). Re-exposed as float32 by
    io-cast-int64-input-float32."""
    tensors = (
        TensorSpec("ids", "int64", (1, 96)),
        TensorSpec("ids_f32", "float32", (1, 96)),
        TensorSpec("output", "float32", (1, 96)),
    )
    ops = (
        OpSpec("CAST", (0,), (1,)),
        OpSpec("ABS", (1,), (2,)),
    )
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(2,)))


def fixture_rank2_add_const() -> bytes:
    """Rank-2 ADD of an activation against a rank-2 CONSTANT — the form one GPU
    delegate computes silently wrong (data/matrix/gpu_metal_mac__2.1.6.json,
    ADD `incorrect`, rank=2 operand_b=constant, measured 2026-08-19 on a ViT
    patch-embedding + learned position table). Rewritten by
    rank2-elementwise-const-to-rank3 into RESHAPE -> ADD(rank 3) -> RESHAPE."""
    n, c = 8, 4
    bias = struct.pack("<%df" % (n * c), *[0.25 * (i % 7) - 0.5 for i in range(n * c)])
    tensors = (
        TensorSpec("tokens", "float32", (n, c)),
        TensorSpec("pos_table", "float32", (n, c), data=bias),
        TensorSpec("output", "float32", (n, c)),
    )
    ops = (OpSpec("ADD", (0, 1), (2,), builtin_options=BuiltinOptionsSpec(type_code=11)),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,), outputs=(2,)))


FIXTURES = {
    "model_select_fixture.tflite": fixture_select(),
    "model_select_v2_fixture.tflite": fixture_select_v2(),
    "model_int64_input_fixture.tflite": fixture_int64_input(),
    "model_rank2_add_const_fixture.tflite": fixture_rank2_add_const(),
}


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    for name, data in FIXTURES.items():
        path = out_dir / name
        path.write_bytes(data)
        print(path)
