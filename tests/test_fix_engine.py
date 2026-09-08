"""Fix engine end-to-end: every v1 action fixes its example fixture, the
re-lint gate rolls back non-improvements, verification rolls back changed
math, determinism and idempotence hold, and reports validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import REPO_ROOT
from litert_compat.cards.schema_io import schema_errors
from litert_compat.fix.engine import run_fix
from litert_compat.fix.rules import load_rule_file, load_rules
from litert_compat.matrix.canonical import canonical_dumps
from litert_compat.matrix.query import Matrix
from litert_compat.parser.builder import (
    BuiltinOptionsSpec,
    GraphSpec,
    OpSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.probe.runners import CpuRunner, Tolerance

EXAMPLES = REPO_ROOT / "data" / "examples"
RULES_DIR = EXAMPLES / "transforms"
MATRIX_PATH = EXAMPLES / "matrix_fix_example.json"

RUNNERS_AVAILABLE = CpuRunner().availability() is None

ACTION_FIXTURES = {
    "replace_op": "model_fix_div_example.tflite",
    "decompose": "model_fix_square_example.tflite",
    "insert_cast": "model_fix_int_mul_example.tflite",
    "io_cast": "model_fix_io64_example.tflite",
}


@pytest.fixture(scope="module")
def matrix() -> Matrix:
    return Matrix.load(MATRIX_PATH)


@pytest.fixture(scope="module")
def rules():
    return load_rules(RULES_DIR)


def fix(model: Path, matrix: Matrix, rules: list, **kwargs: Any):
    defaults: dict[str, Any] = {
        "matrix_path": str(MATRIX_PATH),
        "rules_dir": str(RULES_DIR),
        "mode": "dry_run",
        "tolerance": Tolerance(),
    }
    defaults.update(kwargs)
    return run_fix(model_path=model, matrix=matrix, rules=rules, **defaults)


def write_rule(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / f"{doc['id']}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def tmp_rule(tmp_path: Path, **overrides: Any):
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "id": "tmp-rule",
        "title": "test-only rule",
        "match": {"op": "DIV", "dtypes": ["int32"]},
        "action": "replace_op",
        "params": {"new_op": "FLOOR_DIV"},
        "expected_effect": "test",
        "provenance": "example",
        "fixture": {"model": "m.tflite", "matrix": "x.json"},
    }
    doc.update(overrides)
    return load_rule_file(write_rule(tmp_path, doc))


# --- every v1 action, end-to-end on its example fixture ---------------------


@pytest.mark.parametrize("action", sorted(ACTION_FIXTURES))
def test_action_fixes_its_example_fixture(action: str, matrix: Matrix, rules: list) -> None:
    result = fix(EXAMPLES / ACTION_FIXTURES[action], matrix, rules)
    report = result.report
    assert report["outcome"] == "fixed", report
    assert [i["action"] for i in report["plan"]] == [action]
    assert report["improvement"]["improved"] is True
    assert report["conflicts"] == []
    assert report["unmatched_findings"] == []
    assert report["output"]["sha256"] is not None
    assert result.fixed_bytes is not None
    assert schema_errors(report, "fix_report.schema.json") == []
    # Every applied change traces to a rule id with its evidence (spec §10.1).
    for item in report["plan"]:
        assert item["rule_id"] in report["rules_loaded"]
        assert item["evidence"] is not None
        assert item["provenance"] == "example"


@pytest.mark.parametrize("action", sorted(ACTION_FIXTURES))
@pytest.mark.skipif(not RUNNERS_AVAILABLE, reason="edge-compat[runners] not installed")
def test_action_verification_passes_for_real(
    action: str, matrix: Matrix, rules: list
) -> None:
    result = fix(EXAMPLES / ACTION_FIXTURES[action], matrix, rules)
    verification = result.report["verification"]
    assert verification["status"] == "passed", verification
    assert verification["max_abs_diff"] == 0.0  # the example rewrites are exact
    assert result.report["unverified"] is False


def test_metric_improves_per_action(matrix: Matrix, rules: list) -> None:
    # Graph-surgery actions improve delegated coverage; io_cast improves the
    # interface while keeping lint metrics unregressed.
    for action in ("replace_op", "decompose", "insert_cast"):
        report = fix(EXAMPLES / ACTION_FIXTURES[action], matrix, rules).report
        assert report["before"]["summary"]["coverage_ops_pct"] == 0.0
        assert report["after"]["summary"]["coverage_ops_pct"] == 100.0
    report = fix(EXAMPLES / ACTION_FIXTURES["io_cast"], matrix, rules).report
    assert report["io_changes"] == {
        "before": {"inputs": ["int64"], "outputs": ["int64"]},
        "after": {"inputs": ["int32"], "outputs": ["int32"]},
    }
    assert report["after"]["summary"]["coverage_ops_pct"] == 100.0


# --- determinism and idempotence --------------------------------------------


@pytest.mark.parametrize("action", sorted(ACTION_FIXTURES))
def test_fix_is_deterministic(action: str, matrix: Matrix, rules: list) -> None:
    runs = [fix(EXAMPLES / ACTION_FIXTURES[action], matrix, rules) for _ in range(2)]
    assert runs[0].fixed_bytes == runs[1].fixed_bytes
    assert canonical_dumps(runs[0].report) == canonical_dumps(runs[1].report)


@pytest.mark.parametrize("action", sorted(ACTION_FIXTURES))
def test_fix_on_own_output_applies_zero_changes(
    action: str, matrix: Matrix, rules: list, tmp_path: Path
) -> None:
    first = fix(EXAMPLES / ACTION_FIXTURES[action], matrix, rules)
    fixed_path = tmp_path / "fixed.tflite"
    fixed_path.write_bytes(first.fixed_bytes)
    second = fix(fixed_path, matrix, rules)
    assert second.report["outcome"] == "nothing_to_fix"
    assert second.report["plan"] == []
    assert second.fixed_bytes is None


def test_multi_rule_plan_applies_in_one_run(
    matrix: Matrix, rules: list, tmp_path: Path
) -> None:
    """Two findings, two different rules, one deterministic plan."""
    data = build_tflite(GraphSpec(
        tensors=(
            TensorSpec("a", "int32", (1, 8)),
            TensorSpec("b", "int32", (1, 8)),
            TensorSpec("quot", "int32", (1, 8)),
            TensorSpec("out", "int32", (1, 8)),
        ),
        ops=(
            OpSpec("DIV", (0, 1), (2,), builtin_options=BuiltinOptionsSpec(type_code=29)),
            OpSpec("MUL", (2, 0), (3,), builtin_options=BuiltinOptionsSpec(type_code=21)),
        ),
        inputs=(0, 1),
        outputs=(3,),
    ))
    model = tmp_path / "two_findings.tflite"
    model.write_bytes(data)
    result = fix(model, matrix, rules)
    report = result.report
    assert report["outcome"] == "fixed"
    assert [(i["rule_id"], i["node_index"]) for i in report["plan"]] == [
        ("example-replace-div-floor-div", 0),
        ("example-insert-cast-mul-int32", 1),
    ]
    assert report["after"]["summary"]["coverage_ops_pct"] == 100.0
    if RUNNERS_AVAILABLE:
        assert report["verification"]["status"] == "passed"


# --- rollback paths ----------------------------------------------------------


def test_no_improvement_rolls_back(matrix: Matrix, tmp_path: Path) -> None:
    # MUL[int32] is also fallback in the demo matrix: replacing DIV with MUL
    # applies cleanly but improves nothing — automatic rollback.
    rule = tmp_rule(tmp_path, id="tmp-div-to-mul", params={"new_op": "MUL"})
    result = fix(EXAMPLES / ACTION_FIXTURES["replace_op"], matrix, [rule])
    report = result.report
    assert report["outcome"] == "no_improvement"
    assert report["improvement"]["improved"] is False
    assert report["after"] is not None  # kept for transparency
    assert report["verification"] is None  # rollback happens before verification
    assert result.fixed_bytes is None
    assert report["output"] == {"path": None, "sha256": None, "written": False}
    assert schema_errors(report, "fix_report.schema.json") == []


@pytest.mark.skipif(not RUNNERS_AVAILABLE, reason="edge-compat[runners] not installed")
def test_verification_failure_rolls_back_even_with_allow_unverified(
    matrix: Matrix, tmp_path: Path
) -> None:
    # EXP[float32] is delegated in the demo matrix, so SQUARE → EXP passes the
    # improvement gate — but exp(x) != x², so verification must roll it back.
    rule = tmp_rule(
        tmp_path,
        id="tmp-square-to-exp",
        match={"op": "SQUARE", "dtypes": ["float32"]},
        params={"new_op": "EXP"},
    )
    out = tmp_path / "fixed.tflite"
    result = fix(
        EXAMPLES / ACTION_FIXTURES["decompose"], matrix, [rule],
        mode="apply", allow_unverified=True, out_path=out,
    )
    report = result.report
    assert report["outcome"] == "verification_failed"
    assert report["verification"]["status"] == "failed"
    assert report["verification"]["max_abs_diff"] > 0.1
    assert report["improvement"]["improved"] is True  # the gate passed; numerics failed
    assert result.fixed_bytes is None
    assert not out.exists()
    assert report["unverified"] is False  # a measured failure is never 'unverified'
    assert schema_errors(report, "fix_report.schema.json") == []


# --- matching honesty: conflicts, unmatched, refusals ------------------------


def test_conflicting_rules_leave_node_untouched(matrix: Matrix, tmp_path: Path) -> None:
    rule_a = tmp_rule(tmp_path, id="tmp-a", params={"new_op": "FLOOR_DIV"})
    rule_b = tmp_rule(tmp_path, id="tmp-b", params={"new_op": "MUL"})
    result = fix(EXAMPLES / ACTION_FIXTURES["replace_op"], matrix, [rule_a, rule_b])
    report = result.report
    assert report["outcome"] == "no_rules_matched"
    assert report["plan"] == []
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["rule_ids"] == ["tmp-a", "tmp-b"]
    assert result.fixed_bytes is None
    assert schema_errors(report, "fix_report.schema.json") == []


def test_unmatched_findings_are_reported_never_guessed_at(
    matrix: Matrix, rules: list
) -> None:
    # model_split's findings (GATHER_ND, RESHAPE, SOFTMAX, ...) match none of
    # the example rules under the fix-demo matrix.
    result = fix(EXAMPLES / "model_split_example.tflite", matrix, rules)
    report = result.report
    assert report["outcome"] == "no_rules_matched"
    assert report["plan"] == []
    assert report["unmatched_findings"], "non-delegated findings must be listed"
    for finding in report["unmatched_findings"]:
        assert finding["status"] != "delegated"
    assert schema_errors(report, "fix_report.schema.json") == []


def test_op_sequence_rule_is_refused_with_reason(matrix: Matrix, tmp_path: Path) -> None:
    rule = tmp_rule(
        tmp_path,
        id="tmp-seq",
        match={"op_sequence": ["DIV", "MUL"], "dtypes": ["int32"]},
    )
    result = fix(EXAMPLES / ACTION_FIXTURES["replace_op"], matrix, [rule])
    report = result.report
    assert report["outcome"] == "no_rules_matched"
    assert report["refused"] == [
        {"rule_id": "tmp-seq",
         "reason": "op_sequence matching is not implemented in the v1 engine "
                   "(Deferred); the rule is schema-valid but was not executed"}
    ]


def test_clean_model_is_nothing_to_fix(matrix: Matrix, rules: list, tmp_path: Path) -> None:
    data = build_tflite(GraphSpec(
        tensors=(
            TensorSpec("in", "int64", (1, 4)),
            TensorSpec("out", "int64", (1, 4)),
        ),
        ops=(OpSpec("NEG", (0,), (1,)),),
        inputs=(0,),
        outputs=(1,),
    ))
    # NEG is delegated and no io_cast rule is passed: nothing to fix.
    model = tmp_path / "clean.tflite"
    model.write_bytes(data)
    no_io_rules = [r for r in rules if r.action != "io_cast"]
    result = fix(model, matrix, no_io_rules)
    assert result.report["outcome"] == "nothing_to_fix"
    assert result.report["unmatched_findings"] == []


# --- the --allow-unverified gate ---------------------------------------------


def test_apply_refuses_when_verification_unavailable(
    matrix: Matrix, rules: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "litert_compat.fix.verify.availability", lambda: "ai-edge-litert not importable (test)"
    )
    out = tmp_path / "fixed.tflite"
    result = fix(
        EXAMPLES / ACTION_FIXTURES["replace_op"], matrix, rules,
        mode="apply", out_path=out,
    )
    assert result.refusal is not None and "--allow-unverified" in result.refusal
    assert not out.exists()
    assert result.report["output"]["written"] is False
    assert result.report["verification"]["status"] == "skipped"


def test_apply_with_allow_unverified_marks_report(
    matrix: Matrix, rules: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "litert_compat.fix.verify.availability", lambda: "ai-edge-litert not importable (test)"
    )
    out = tmp_path / "fixed.tflite"
    result = fix(
        EXAMPLES / ACTION_FIXTURES["replace_op"], matrix, rules,
        mode="apply", allow_unverified=True, out_path=out,
    )
    report = result.report
    assert result.refusal is None
    assert report["outcome"] == "fixed"
    assert report["unverified"] is True
    assert report["verification"]["status"] == "skipped"
    assert out.exists() and out.read_bytes() == result.fixed_bytes
    assert report["output"]["written"] is True
    assert schema_errors(report, "fix_report.schema.json") == []


# --- real-converter fidelity through the engine ------------------------------


def test_io_cast_remaps_signature_tensor_indices(
    matrix: Matrix, rules: list, tmp_path: Path
) -> None:
    from dataclasses import replace

    from litert_compat.fix.loader import load_graph_spec
    from litert_compat.parser.builder import SignatureDefSpec, TensorMapSpec

    spec = load_graph_spec((EXAMPLES / ACTION_FIXTURES["io_cast"]).read_bytes())
    signature = SignatureDefSpec(
        "serving_default",
        inputs=tuple(TensorMapSpec(f"in{i}", t) for i, t in enumerate(spec.inputs)),
        outputs=tuple(TensorMapSpec(f"out{i}", t) for i, t in enumerate(spec.outputs)),
    )
    model = tmp_path / "signed.tflite"
    model.write_bytes(build_tflite(replace(spec, signature_defs=(signature,))))

    result = fix(model, matrix, rules)
    assert result.report["outcome"] == "fixed"
    fixed = load_graph_spec(result.fixed_bytes)
    remapped = fixed.signature_defs[0]
    assert remapped.signature_key == "serving_default"
    # A rewired graph boundary carries its signature entry along — the maps
    # must point at the new boundary tensors, not the pre-cast originals.
    assert [m.tensor_index for m in remapped.inputs] == list(fixed.inputs)
    assert [m.tensor_index for m in remapped.outputs] == list(fixed.outputs)
    assert [m.name for m in remapped.inputs] == [f"in{i}" for i in range(len(fixed.inputs))]


def test_cast_created_tensors_drop_quantization(tmp_path: Path) -> None:
    from litert_compat.fix.engine import PlanItem, apply_plan
    from litert_compat.parser.builder import QuantizationSpec

    rule = tmp_rule(
        tmp_path,
        id="cast-rule",
        action="insert_cast",
        match={"op": "MUL", "dtypes": ["int32"]},
        params={"to": "float32"},
    )
    quant = QuantizationSpec(scale=(0.5,), zero_point=(1,))
    graph = GraphSpec(
        tensors=(
            TensorSpec("a", "int32", (1, 4), quantization=quant),
            TensorSpec("b", "int32", (1, 4), quantization=quant),
            TensorSpec("out", "int32", (1, 4), quantization=quant),
        ),
        ops=(OpSpec("MUL", (0, 1), (2,)),),
        inputs=(0, 1),
        outputs=(2,),
    )
    fixed = apply_plan(graph, [PlanItem(rule=rule, node_index=0, op="MUL")])
    created = [t for t in fixed.tensors if t.name.startswith("fix__")]
    assert created, "insert_cast must create cast tensors"
    assert all(t.quantization is None for t in created)
    assert all(t.dtype == "float32" for t in created)
    # The original tensors keep their quantization untouched.
    assert all(
        t.quantization == quant for t in fixed.tensors if not t.name.startswith("fix__")
    )
