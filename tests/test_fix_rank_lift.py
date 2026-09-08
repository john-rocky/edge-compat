"""v1.1 engine additions, end-to-end on the committed rank-2 ADD rule:

- the reader marks constant tensors (`TensorInfo.is_constant`), classify exposes
  operand kinds as non-reported `operand_meta`, and matrix/rule constraints such
  as `operand_b=constant` match against it;
- `unsqueeze0:` intermediate shapes and engine-synthesized RESHAPE shape
  constants produce a runnable graph that is bit-exact on the CPU reference;
- the guards refuse a dynamic source shape and a RESHAPE recipe node with more
  than one input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO_ROOT

from litert_compat.fix.engine import run_fix
from litert_compat.fix.rules import load_rule_file
from litert_compat.lint.classify import classify_model, operand_meta
from litert_compat.matrix.query import Matrix
from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    OpSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.parser.reader import parse_tflite
from litert_compat.probe.runners import CpuRunner, Tolerance

TRANSFORMS = REPO_ROOT / "data" / "transforms"
RULE = TRANSFORMS / "rank2-elementwise-const-to-rank3.json"
FIXTURE_MODEL = TRANSFORMS / "fixtures" / "model_rank2_add_const_fixture.tflite"
FIXTURE_MATRIX = TRANSFORMS / "fixtures" / "matrix_gpu_rank2_add_fixture.json"
RUNNERS_AVAILABLE = CpuRunner().availability() is None


def _rank2_add(const_b: bool, dynamic: bool = False) -> bytes:
    n, c = 8, 4
    data = (b"\x00\x00\x80\x3f" * (n * c)) if const_b else None  # 1.0f
    sig = (-1, c) if dynamic else None
    tensors = (
        TensorSpec("x", "float32", (n, c), shape_signature=sig),
        TensorSpec("b", "float32", (n, c), data=data),
        TensorSpec("y", "float32", (n, c), shape_signature=sig),
    )
    ops = (OpSpec("ADD", (0, 1), (2,), builtin_options=BuiltinOptionsSpec(type_code=11)),)
    return build_tflite(GraphSpec(tensors, ops, inputs=(0,) if const_b else (0, 1), outputs=(2,)))


def test_reader_marks_constant_tensors() -> None:
    parsed = parse_tflite(_rank2_add(const_b=True))
    kinds = [t.is_constant for t in parsed.subgraphs[0].tensors]
    assert kinds == [False, True, False]
    sg = parsed.subgraphs[0]
    assert operand_meta(sg, sg.nodes[0]) == {"operand_a": "activation", "operand_b": "constant"}


def test_operand_constraint_matches_only_the_constant_form() -> None:
    matrix = Matrix.load(FIXTURE_MATRIX)
    const = classify_model(parse_tflite(_rank2_add(const_b=True)), matrix)
    act = classify_model(parse_tflite(_rank2_add(const_b=False)), matrix)
    assert const[0].verdict.status == "incorrect"
    # activation + activation at rank 2: no entry matches (operand_b differs) -> unknown,
    # and the reported shape_meta does not carry operand keys (reports stay pinned).
    assert act[0].verdict.status == "unknown"
    assert "operand_b" not in const[0].shape_meta
    assert const[0].match_meta["operand_b"] == "constant"


def _fix(model_bytes: bytes, tmp_path: Path, **kw: Any):
    model = tmp_path / "m.tflite"
    model.write_bytes(model_bytes)
    rule = load_rule_file(RULE)
    matrix = Matrix.load(FIXTURE_MATRIX)
    defaults = {"matrix_path": str(FIXTURE_MATRIX), "rules_dir": str(TRANSFORMS),
                "mode": "dry_run", "tolerance": Tolerance()}
    defaults.update(kw)
    return run_fix(model_path=model, matrix=matrix, rules=[rule], **defaults).report


@pytest.mark.skipif(not RUNNERS_AVAILABLE, reason="CPU runner extra not installed")
def test_rank_lift_rule_fixes_fixture_bit_exact(tmp_path: Path) -> None:
    report = _fix(FIXTURE_MODEL.read_bytes(), tmp_path)
    assert report["outcome"] == "fixed", json.dumps(report, indent=1)[:2000]
    assert report["before"]["summary"]["status_counts"] == {"incorrect": 1}
    assert report["after"]["summary"]["status_counts"] == {"delegated": 4}
    assert report["plan"][0]["detail"] == "ADD → RESHAPE + RESHAPE + ADD + RESHAPE"
    assert report["verification"]["status"] == "passed"
    assert report["verification"]["max_abs_diff"] == 0.0


def test_rank_lift_refuses_dynamic_source_shape(tmp_path: Path) -> None:
    report = _fix(_rank2_add(const_b=True, dynamic=True), tmp_path)
    assert report["outcome"] != "fixed"
    reasons = " ".join(r["reason"] for r in report["refused"])
    assert "dynamic" in reasons


def test_reshape_recipe_node_must_name_one_input(tmp_path: Path) -> None:
    doc = json.loads(RULE.read_text())
    doc["id"] = "bad-reshape"
    doc["params"]["nodes"][0]["inputs"] = ["in:0", "in:1"]
    path = tmp_path / "bad-reshape.json"
    path.write_text(json.dumps(doc))
    rule = load_rule_file(path)
    model = tmp_path / "m.tflite"
    model.write_bytes(FIXTURE_MODEL.read_bytes())
    report = run_fix(model_path=model, matrix=Matrix.load(FIXTURE_MATRIX), rules=[rule],
                     matrix_path=str(FIXTURE_MATRIX), rules_dir=str(tmp_path),
                     mode="dry_run", tolerance=Tolerance()).report
    assert report["outcome"] != "fixed"
    assert any("exactly one" in r["reason"] for r in report["refused"])
