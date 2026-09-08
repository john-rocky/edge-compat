"""The fix engine: lint → match → plan → apply → re-lint gate → verify.

Safety contract (spec §10.3), enforced here:
- The input file is never modified; every rewrite happens on a loaded copy.
- The plan is deterministic and conflict-checked: a target claimed by more
  than one rule is left untouched and reported — never resolved by order of
  appearance.
- After applying, the graph is re-linted against the SAME matrix snapshot;
  the metric must improve (delegated coverage, then incorrect+crash count,
  then partition count; io_cast plans may instead reduce the interface
  violation count without regressing the lint metrics) or everything is
  rolled back with a `no_improvement` report.
- Numerical verification runs original vs fixed on the CPU interpreter; a
  measured failure rolls back and is never overridable.
- Unmatched findings are reported, never guessed at.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from litert_compat.fix import verify
from litert_compat.fix.loader import load_graph_spec
from litert_compat.fix.options_registry import default_options_for
from litert_compat.fix.rules import TransformRule
from litert_compat.lint.classify import ClassifiedNode, classify_model
from litert_compat.lint.report import build_report
from litert_compat.matrix.query import Matrix, constraints_match, dtype_match
from litert_compat.parser.builder import (
    GraphSpec,
    MetadataSpec,
    OpSpec,
    SignatureDefSpec,
    TensorSpec,
    build_tflite,
)
from litert_compat.parser.reader import ParsedModel, parse_tflite
from litert_compat.probe.runners import Tolerance

FIX_REPORT_SCHEMA_VERSION = "1.0"


class FixEngineError(ValueError):
    """A usage/data error that prevents the pipeline from running (exit 2)."""


@dataclass(frozen=True)
class IoTarget:
    direction: str  # "input" | "output"
    position: int  # index into graph inputs/outputs
    tensor_index: int
    name: str | None
    dtype: str


@dataclass(frozen=True)
class PlanItem:
    rule: TransformRule
    node_index: int | None = None  # original-graph node index; None for io_cast
    op: str | None = None
    io_targets: tuple[IoTarget, ...] = ()

    def detail(self) -> str:
        if self.rule.action == "replace_op":
            return f"{self.op} → {self.rule.params['new_op']}"
        if self.rule.action == "decompose":
            ops = " + ".join(n["op"] for n in self.rule.params["nodes"])
            return f"{self.op} → {ops}"
        if self.rule.action == "insert_cast":
            return f"compute {self.op} in {self.rule.params['to']}"
        wrapped = ", ".join(
            f"{t.direction} {t.name or t.tensor_index} ({t.dtype})" for t in self.io_targets
        )
        return f"graph I/O → {self.rule.params['to']}: {wrapped}"


@dataclass(frozen=True)
class FixResult:
    report: dict[str, Any]
    fixed_bytes: bytes | None
    refusal: str | None  # --apply refused (verification unavailable); exit 2


@dataclass
class _MutableGraph:
    tensors: list[TensorSpec]
    ops: list[OpSpec]
    inputs: list[int]
    outputs: list[int]
    description: str
    name: str
    metadata: tuple[MetadataSpec, ...]
    signature_defs: list[SignatureDefSpec]

    @classmethod
    def from_spec(cls, spec: GraphSpec) -> _MutableGraph:
        return cls(
            tensors=list(spec.tensors),
            ops=list(spec.ops),
            inputs=list(spec.inputs),
            outputs=list(spec.outputs),
            description=spec.description,
            name=spec.name,
            metadata=spec.metadata,
            signature_defs=list(spec.signature_defs),
        )

    def to_spec(self) -> GraphSpec:
        return GraphSpec(
            tensors=tuple(self.tensors),
            ops=tuple(self.ops),
            inputs=tuple(self.inputs),
            outputs=tuple(self.outputs),
            description=self.description,
            name=self.name,
            metadata=self.metadata,
            signature_defs=tuple(self.signature_defs),
        )

    def add_tensor(self, spec: TensorSpec) -> int:
        self.tensors.append(spec)
        return len(self.tensors) - 1


@dataclass
class _MatchState:
    plan: list[PlanItem] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    matched_nodes: set[int] = field(default_factory=set)


def _fixable(classified: list[ClassifiedNode]) -> list[ClassifiedNode]:
    """Everything not verdicted `delegated` is fixable — including claimed but
    imperfect verdicts (`partial`, `incorrect`): fixing an incorrect op is the
    whole point of the incorrect+crash component of the improvement gate."""
    return [c for c in classified if c.verdict.status != "delegated"]


def _io_meta(graph: ParsedModel, tensor_index: int) -> dict[str, Any]:
    tensor = graph.subgraphs[0].tensors[tensor_index]
    meta: dict[str, Any] = {"dynamic_shape": tensor.is_dynamic}
    if tensor.rank is not None:
        meta["rank"] = tensor.rank
    return meta


def _io_targets(parsed: ParsedModel, rule: TransformRule) -> tuple[IoTarget, ...]:
    subgraph = parsed.subgraphs[0]
    scope = rule.params.get("io", "both")
    dtypes = set(rule.match["dtypes"])
    constraints = rule.match.get("constraints")
    targets: list[IoTarget] = []
    scoped = []
    if scope in ("inputs", "both"):
        scoped.append(("input", subgraph.inputs))
    if scope in ("outputs", "both"):
        scoped.append(("output", subgraph.outputs))
    for direction, indices in scoped:
        for position, tensor_index in enumerate(indices):
            tensor = subgraph.tensors[tensor_index]
            if tensor.dtype not in dtypes:
                continue
            ok, _ = constraints_match(constraints, _io_meta(parsed, tensor_index))
            if not ok:
                continue
            targets.append(
                IoTarget(direction, position, tensor_index, tensor.name, tensor.dtype)
            )
    return tuple(targets)


def _decompose_plan_error(
    rule: TransformRule, node: OpSpec, parsed: ParsedModel | None = None
) -> str | None:
    """Recipe checks that need the matched node's arity (and, for RESHAPE
    recipes, the matched tensors' shapes)."""
    tensors = parsed.subgraphs[0].tensors if parsed is not None else None
    for i, spec in enumerate(rule.params["nodes"]):
        if spec["op"] == "RESHAPE":
            if len(spec["inputs"]) != 1:
                return (f"recipe nodes[{i}] RESHAPE must name exactly one (data) input; "
                        "the engine synthesizes the shape input from the output tensor")
            out_ref = spec["outputs"][0]
            kind, _, index = out_ref.partition(":")
            if kind == "out" and tensors is not None and int(index) < len(node.outputs):
                t = tensors[node.outputs[int(index)]]
                if t.shape is None or t.is_dynamic:
                    return (f"recipe nodes[{i}] RESHAPE writes {out_ref}, whose shape is "
                            "dynamic/unknown — a static target shape is required")
            # tmp: outputs derive their shape from a matched tensor via like:/unsqueeze0:
            # and are checked through that tensor below.
        for ref in (*spec["inputs"], *spec["outputs"]):
            kind, _, index = ref.partition(":")
            if kind == "in" and int(index) >= len(node.inputs):
                return (f"recipe nodes[{i}] references {ref} but the matched node "
                        f"has {len(node.inputs)} inputs")
            if kind == "in" and node.inputs[int(index)] == -1:
                return (f"recipe nodes[{i}] references {ref} but the matched node's "
                        f"input {index} is optional-absent (-1)")
            if kind == "out" and int(index) >= len(node.outputs):
                return (f"recipe nodes[{i}] references {ref} but the matched node "
                        f"has {len(node.outputs)} outputs")
    produced = {int(r.partition(":")[2]) for s in rule.params["nodes"]
                for r in s["outputs"] if r.startswith("out:")}
    missing = sorted(set(range(len(node.outputs))) - produced)
    if missing:
        return ("recipe does not produce matched-node output(s) "
                + ", ".join(f"out:{k}" for k in missing))
    reshape_tmp_outputs = {r.partition(":")[2] for sp in rule.params["nodes"]
                           if sp["op"] == "RESHAPE" for r in sp["outputs"] if r.startswith("tmp:")}
    for i, spec in enumerate(rule.params.get("intermediates") or []):
        for key in ("dtype", "shape"):
            value = spec[key]
            if isinstance(value, str) and value.startswith(("like:", "unsqueeze0:")):
                mode, kind, index = value.split(":")
                bound = len(node.inputs) if kind == "in" else len(node.outputs)
                if int(index) >= bound:
                    return (f"intermediates[{i}].{key} references {value} but the "
                            f"matched node has {bound} {kind}put tensors")
                if kind == "in" and node.inputs[int(index)] == -1:
                    return (f"intermediates[{i}].{key} references {value} but the "
                            f"matched node's input {index} is optional-absent (-1)")
                if key == "shape" and tensors is not None and (
                    mode == "unsqueeze0" or spec["name"] in reshape_tmp_outputs
                ):
                    pool = node.inputs if kind == "in" else node.outputs
                    t = tensors[pool[int(index)]]
                    if t.shape is None or t.is_dynamic:
                        return (f"intermediates[{i}].shape derives from {value}, whose shape "
                                "is dynamic/unknown — derived shapes need a static source")
    return None


def _match_rules(
    parsed: ParsedModel,
    classified: list[ClassifiedNode],
    rules: list[TransformRule],
) -> _MatchState:
    state = _MatchState()
    fixable = _fixable(classified)

    node_claims: dict[int, list[TransformRule]] = {}
    io_claims: dict[tuple[str, int], list[tuple[TransformRule, IoTarget]]] = {}
    for rule in rules:
        if rule.uses_op_sequence:
            state.refused.append({
                "rule_id": rule.id,
                "reason": "op_sequence matching is not implemented in the v1 engine "
                          "(Deferred); the rule is schema-valid but was not executed",
            })
            continue
        if rule.action == "io_cast":
            for target in _io_targets(parsed, rule):
                io_claims.setdefault((target.direction, target.position), []).append(
                    (rule, target)
                )
            continue
        for c in fixable:
            # The loader rewrites single-subgraph models only; findings in
            # other subgraphs stay reported under unmatched_findings.
            if c.subgraph != 0 or c.is_custom or c.node.op != rule.match.get("op"):
                continue
            dtype_ok, _ = dtype_match(rule.match.get("dtypes"), c.dtypes)
            constraints_ok, _ = constraints_match(rule.match.get("constraints"), c.match_meta)
            if dtype_ok and constraints_ok:
                node_claims.setdefault(c.node.index, []).append(rule)

    nodes_by_index = {c.node.index: c for c in classified}
    for node_index in sorted(node_claims):
        claimants = node_claims[node_index]
        node = nodes_by_index[node_index].node
        if len(claimants) > 1:
            state.conflicts.append({
                "subgraph": 0,
                "node_index": node_index,
                "op": node.op,
                "rule_ids": sorted(r.id for r in claimants),
                "reason": "node claimed by more than one rule — left untouched",
            })
            continue
        rule = claimants[0]
        if rule.action == "decompose":
            spec_node = OpSpec(node.op, node.inputs, node.outputs)
            error = _decompose_plan_error(rule, spec_node, parsed)
            if error:
                state.refused.append({
                    "rule_id": rule.id,
                    "reason": f"node {node_index} ({node.op}): {error}",
                })
                continue
        state.plan.append(PlanItem(rule=rule, node_index=node_index, op=node.op))
        state.matched_nodes.add(node_index)

    io_rules: dict[str, list[IoTarget]] = {}
    io_rule_by_id: dict[str, TransformRule] = {}
    for key in sorted(io_claims):
        claimants = io_claims[key]
        if len({rule.id for rule, _ in claimants}) > 1:
            target = claimants[0][1]
            state.conflicts.append({
                "tensor_index": target.tensor_index,
                "rule_ids": sorted({rule.id for rule, _ in claimants}),
                "reason": f"graph {target.direction} tensor claimed by more than one "
                          "io_cast rule — left untouched",
            })
            continue
        rule, target = claimants[0]
        io_rules.setdefault(rule.id, []).append(target)
        io_rule_by_id[rule.id] = rule
    for rule_id in sorted(io_rules):
        state.plan.append(PlanItem(
            rule=io_rule_by_id[rule_id],
            io_targets=tuple(sorted(io_rules[rule_id],
                                    key=lambda t: (t.direction, t.position))),
        ))
    return state


def _fresh_name(rule_id: str, suffix: str) -> str:
    return f"fix__{rule_id.replace('-', '_')}__{suffix}"


def _apply_replace_op(g: _MutableGraph, item: PlanItem) -> None:
    node = g.ops[item.node_index or 0]
    params = item.rule.params
    g.ops[item.node_index or 0] = OpSpec(
        op=params["new_op"],
        inputs=node.inputs,
        outputs=node.outputs,
        version=params.get("op_version", 1),
        builtin_options=default_options_for(params["new_op"]),
    )


def _like_tensor(g: _MutableGraph, node: OpSpec, ref: str) -> TensorSpec:
    """Resolve a shape/dtype reference. `like:in|out:K` borrows the tensor as-is;
    `unsqueeze0:in|out:K` borrows it with a leading unit axis prepended (rank+1)
    — the only derived shape the engine knows, enough to lift a rank-2 operand to
    the rank-3 form a delegate computes correctly."""
    mode, kind, index = ref.split(":")
    pool = node.inputs if kind == "in" else node.outputs
    tensor = g.tensors[pool[int(index)]]
    if mode == "like":
        return tensor
    if mode == "unsqueeze0":
        sig = tensor.shape_signature
        return replace(
            tensor,
            shape=(1, *tensor.shape),
            shape_signature=(1, *sig) if sig is not None else None,
        )
    raise ValueError(f"unknown shape reference {ref!r}")


def _apply_decompose(g: _MutableGraph, item: PlanItem) -> None:
    n = item.node_index or 0
    node = g.ops[n]
    params = item.rule.params
    tmp_index: dict[str, int] = {}
    for spec in params.get("intermediates") or []:
        dtype = spec["dtype"]
        if isinstance(dtype, str) and dtype.startswith("like:"):
            dtype = _like_tensor(g, node, dtype).dtype
        like = _like_tensor(g, node, spec["shape"])
        tmp_index[spec["name"]] = g.add_tensor(TensorSpec(
            name=_fresh_name(item.rule.id, f"n{n}_{spec['name']}"),
            dtype=dtype,
            shape=like.shape,
            shape_signature=like.shape_signature,
        ))

    def resolve(ref: str) -> int:
        kind, _, index = ref.partition(":")
        if kind == "in":
            return node.inputs[int(index)]
        if kind == "out":
            return node.outputs[int(index)]
        return tmp_index[index]

    recipe = []
    for j, spec in enumerate(params["nodes"]):
        inputs = [resolve(r) for r in spec["inputs"]]
        outputs = tuple(resolve(r) for r in spec["outputs"])
        if spec["op"] == "RESHAPE" and len(inputs) == 1:
            # RESHAPE takes its target shape from a second, constant int32 input.
            # The recipe names only the data input; the engine synthesizes the
            # shape vector from the declared OUTPUT tensor (static shape required
            # — refused upstream by _decompose_plan_error), so no rule ever
            # carries a freeform shape literal.
            out_shape = g.tensors[outputs[0]].shape
            shape_tensor = g.add_tensor(TensorSpec(
                name=_fresh_name(item.rule.id, f"n{n}_r{j}_shape"),
                dtype="int32",
                shape=(len(out_shape),),
                data=b"".join(int(d).to_bytes(4, "little", signed=True) for d in out_shape),
            ))
            inputs.append(shape_tensor)
        recipe.append(OpSpec(
            op=spec["op"],
            inputs=tuple(inputs),
            outputs=outputs,
            builtin_options=default_options_for(spec["op"]),
        ))
    g.ops[n : n + 1] = recipe


def _apply_insert_cast(g: _MutableGraph, item: PlanItem) -> None:
    n = item.node_index or 0
    node = g.ops[n]
    to = item.rule.params["to"]
    cast_options = default_options_for("CAST")

    before: list[OpSpec] = []
    remapped: dict[int, int] = {}
    for k, tensor_index in enumerate(dict.fromkeys(node.inputs)):
        if tensor_index == -1:
            continue  # optional input deliberately absent — nothing to cast
        tensor = g.tensors[tensor_index]
        if tensor.dtype == to:
            continue
        cast_index = g.add_tensor(replace(
            tensor, name=_fresh_name(item.rule.id, f"n{n}_in{k}"), dtype=to, data=None,
            quantization=None,
        ))
        before.append(OpSpec("CAST", (tensor_index,), (cast_index,),
                             builtin_options=cast_options))
        remapped[tensor_index] = cast_index

    after: list[OpSpec] = []
    new_outputs: list[int] = []
    for k, tensor_index in enumerate(node.outputs):
        tensor = g.tensors[tensor_index]
        if tensor.dtype == to:
            new_outputs.append(tensor_index)
            continue
        mid_index = g.add_tensor(replace(
            tensor, name=_fresh_name(item.rule.id, f"n{n}_out{k}"), dtype=to, data=None,
            quantization=None,
        ))
        after.append(OpSpec("CAST", (mid_index,), (tensor_index,),
                            builtin_options=cast_options))
        new_outputs.append(mid_index)

    rewired = replace(
        node,
        inputs=tuple(remapped.get(t, t) for t in node.inputs),
        outputs=tuple(new_outputs),
    )
    g.ops[n : n + 1] = [*before, rewired, *after]


def _apply_io_cast(g: _MutableGraph, item: PlanItem) -> None:
    to = item.rule.params["to"]
    cast_options = default_options_for("CAST")
    prepend: list[OpSpec] = []
    input_remap: dict[int, int] = {}
    output_remap: dict[int, int] = {}
    for target in item.io_targets:
        tensor_index = (g.inputs if target.direction == "input" else g.outputs)[target.position]
        tensor = g.tensors[tensor_index]
        boundary_index = g.add_tensor(replace(
            tensor,
            name=_fresh_name(item.rule.id, f"io_{target.direction}{target.position}"),
            dtype=to,
            data=None,
            quantization=None,
        ))
        if target.direction == "input":
            prepend.append(OpSpec("CAST", (boundary_index,), (tensor_index,),
                                  builtin_options=cast_options))
            g.inputs[target.position] = boundary_index
            input_remap[tensor_index] = boundary_index
        else:
            g.ops.append(OpSpec("CAST", (tensor_index,), (boundary_index,),
                                builtin_options=cast_options))
            g.outputs[target.position] = boundary_index
            output_remap[tensor_index] = boundary_index
    g.ops[0:0] = prepend
    # Signature defs address graph I/O by tensor index; a rewired boundary
    # must carry its signature entry along, or a signature runner would feed
    # or fetch the pre-cast tensor and bypass the fix.
    if g.signature_defs and (input_remap or output_remap):
        g.signature_defs = [
            replace(
                sig,
                inputs=tuple(
                    replace(m, tensor_index=input_remap.get(m.tensor_index, m.tensor_index))
                    for m in sig.inputs
                ),
                outputs=tuple(
                    replace(m, tensor_index=output_remap.get(m.tensor_index, m.tensor_index))
                    for m in sig.outputs
                ),
            )
            for sig in g.signature_defs
        ]


_ACTIONS = {
    "replace_op": _apply_replace_op,
    "decompose": _apply_decompose,
    "insert_cast": _apply_insert_cast,
    "io_cast": _apply_io_cast,
}


def apply_plan(spec: GraphSpec, plan: list[PlanItem]) -> GraphSpec:
    """Apply plan items on a mutable copy: node items in descending node order
    (so original indices stay valid), io_cast items last. Deterministic."""
    g = _MutableGraph.from_spec(spec)
    node_items = sorted(
        (i for i in plan if i.node_index is not None),
        key=lambda i: i.node_index or 0,
        reverse=True,
    )
    for item in node_items:
        _ACTIONS[item.rule.action](g, item)
    for item in sorted((i for i in plan if i.node_index is None), key=lambda i: i.rule.id):
        _ACTIONS[item.rule.action](g, item)
    return g.to_spec()


def _lint(model_bytes: bytes, matrix: Matrix, model_path: str, matrix_path: str,
          parsed: ParsedModel | None = None) -> dict[str, Any]:
    parsed = parsed or parse_tflite(model_bytes)
    classified = classify_model(parsed, matrix)
    return build_report(
        classified, matrix,
        model_path=model_path, model_bytes=model_bytes,
        subgraph_count=len(parsed.subgraphs), matrix_path=matrix_path,
    )


def _ic_count(summary: dict[str, Any]) -> int:
    counts = summary["status_counts"]
    return counts.get("incorrect", 0) + counts.get("crash", 0)


def improvement_verdict(
    before: dict[str, Any], after: dict[str, Any], io_before: int, io_after: int
) -> tuple[bool, str]:
    """The re-lint gate. Lexicographic on the lint metrics: delegated coverage
    first, then incorrect+crash count, then partition count. An io_cast plan
    may instead reduce the interface-violation count, provided the lint
    metrics do not regress."""
    cov_b, cov_a = before["coverage_ops_pct"], after["coverage_ops_pct"]
    ic_b, ic_a = _ic_count(before), _ic_count(after)
    part_b, part_a = before["partition_count"], after["partition_count"]

    if cov_a > cov_b:
        return True, f"delegated coverage improved: {cov_b}% → {cov_a}%"
    if cov_a == cov_b and ic_a < ic_b:
        return True, f"incorrect+crash verdicts reduced: {ic_b} → {ic_a}"
    if cov_a == cov_b and ic_a == ic_b and part_a < part_b:
        return True, f"partition count reduced: {part_b} → {part_a}"

    regressed = (
        cov_a < cov_b
        or (cov_a == cov_b and ic_a > ic_b)
        or (cov_a == cov_b and ic_a == ic_b and part_a > part_b)
    )
    if io_after < io_before and not regressed:
        return True, (f"graph I/O interface violations reduced: {io_before} → {io_after} "
                      "(lint metrics not regressed)")
    return False, (
        f"no metric improved: coverage {cov_b}% → {cov_a}%, incorrect+crash "
        f"{ic_b} → {ic_a}, partitions {part_b} → {part_a}"
        + (f", I/O violations {io_before} → {io_after}" if io_before or io_after else "")
    )


def _io_dtypes_block(parsed: ParsedModel) -> dict[str, list[str]]:
    subgraph = parsed.subgraphs[0]
    return {
        "inputs": [subgraph.tensors[i].dtype for i in subgraph.inputs],
        "outputs": [subgraph.tensors[i].dtype for i in subgraph.outputs],
    }


def _plan_item_record(item: PlanItem) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rule_id": item.rule.id,
        "action": item.rule.action,
        "provenance": item.rule.provenance,
        "detail": item.detail(),
        "evidence": item.rule.evidence,
        "notes": item.rule.notes,
        "io_targets": [
            {
                "direction": t.direction,
                "tensor_index": t.tensor_index,
                "name": t.name,
                "dtype": t.dtype,
            }
            for t in item.io_targets
        ],
    }
    if item.node_index is not None:
        record["subgraph"] = 0
        record["node_index"] = item.node_index
        record["op"] = item.op
    else:
        record["node_index"] = None
        record["op"] = None
    return record


def _unmatched(classified: list[ClassifiedNode], matched: set[int]) -> list[dict[str, Any]]:
    return [
        {
            "subgraph": c.subgraph,
            "node_index": c.node.index,
            "op": c.node.op,
            "status": c.verdict.status,
            "dtypes": c.dtypes,
        }
        for c in _fixable(classified)
        if c.node.index not in matched
    ]


def run_fix(
    model_path: Path,
    matrix: Matrix,
    matrix_path: str,
    rules: list[TransformRule],
    rules_dir: str,
    mode: str,
    tolerance: Tolerance,
    allow_unverified: bool = False,
    out_path: Path | None = None,
) -> FixResult:
    """Run the full pipeline. In `apply` mode the fixed model is written to
    `out_path` only when the improvement gate passed AND verification passed
    (or was explicitly waived with allow_unverified for skipped/error — a
    measured failure is never waived)."""
    model_bytes = model_path.read_bytes()
    parsed = parse_tflite(model_bytes)
    classified = classify_model(parsed, matrix)
    before_report = _lint(model_bytes, matrix, str(model_path), matrix_path, parsed=parsed)

    state = _match_rules(parsed, classified, rules)
    report: dict[str, Any] = {
        "schema_version": FIX_REPORT_SCHEMA_VERSION,
        "backend": matrix.backend,
        "litert_version": matrix.litert_version,
        "model": {
            "path": str(model_path),
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
        },
        "matrix": {
            "file": matrix_path,
            "litert_version": matrix.litert_version,
            "generated_at": matrix.doc["generated_at"],
        },
        "rules_dir": rules_dir,
        "rules_loaded": [r.id for r in rules],
        "mode": mode,
        "plan": [_plan_item_record(i) for i in state.plan],
        "conflicts": state.conflicts,
        "refused": state.refused,
        "unmatched_findings": _unmatched(classified, state.matched_nodes),
        "io_changes": None,
        "before": {"summary": before_report["summary"]},
        "after": None,
        "improvement": None,
        "verification": None,
        "unverified": False,
        "output": {"path": None, "sha256": None, "written": False},
    }

    if not state.plan:
        report["outcome"] = (
            "nothing_to_fix" if not report["unmatched_findings"] and not state.conflicts
            else "no_rules_matched"
        )
        return FixResult(report=report, fixed_bytes=None, refusal=None)

    try:
        spec = load_graph_spec(model_bytes)
    except Exception as exc:
        raise FixEngineError(f"{model_path}: {exc}") from exc
    fixed_spec = apply_plan(spec, state.plan)
    fixed_bytes = build_tflite(fixed_spec)
    fixed_parsed = parse_tflite(fixed_bytes)
    after_report = _lint(fixed_bytes, matrix, str(model_path), matrix_path,
                         parsed=fixed_parsed)
    report["after"] = {"summary": after_report["summary"]}

    io_items = [i for i in state.plan if i.rule.action == "io_cast"]
    io_before = sum(len(i.io_targets) for i in io_items)
    io_after = sum(len(_io_targets(fixed_parsed, i.rule)) for i in io_items)
    if io_items:
        report["io_changes"] = {
            "before": _io_dtypes_block(parsed),
            "after": _io_dtypes_block(fixed_parsed),
        }

    improved, reason = improvement_verdict(
        before_report["summary"], after_report["summary"], io_before, io_after
    )
    report["improvement"] = {"improved": improved, "reason": reason}
    if not improved:
        report["outcome"] = "no_improvement"
        return FixResult(report=report, fixed_bytes=None, refusal=None)

    unavailable = verify.availability()
    if unavailable is None:
        verification = verify.run_verification(model_path, model_bytes, fixed_bytes, tolerance)
    else:
        verification = verify.skipped_block(unavailable, tolerance)
    report["verification"] = verification

    if verification["status"] == "failed":
        report["outcome"] = "verification_failed"
        return FixResult(report=report, fixed_bytes=None, refusal=None)

    report["outcome"] = "fixed"
    report["unverified"] = verification["status"] != "passed"
    report["output"]["sha256"] = hashlib.sha256(fixed_bytes).hexdigest()

    refusal: str | None = None
    if mode == "apply":
        if report["unverified"] and not allow_unverified:
            refusal = (
                f"refusing --apply: numerical verification is {verification['status']} "
                f"({verification['reason']}); install edge-compat[runners] or pass "
                "--allow-unverified to apply anyway (the report is marked unverified)"
            )
        elif out_path is not None:
            out_path.write_bytes(fixed_bytes)
            report["output"]["path"] = str(out_path)
            report["output"]["written"] = True
    return FixResult(report=report, fixed_bytes=fixed_bytes, refusal=refusal)
