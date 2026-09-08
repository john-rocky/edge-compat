"""Lint report: one internal object (the `--json` shape, validating against
`schemas/lint_report.schema.json`), rendered to text and markdown from it.

Rewrite suggestions are surfaced verbatim from matched matrix entries — never
generated. Every report carries the snapshot's backend + litert_version, the
model hash, and the matrix file identity, so no output can present one
backend's result as another's.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

from litert_compat.lint.classify import ClassifiedNode
from litert_compat.lint.partition import (
    IMPACT_NOTE,
    rank_blocking_ops,
    simulate_partitions,
)
from litert_compat.matrix.query import Matrix

SCHEMA_VERSION = "1.1"  # 1.1: rewrite-suggestion hints may carry transform_id (matrix 1.2)


def load_lint_report_schema() -> dict[str, Any]:
    """Load lint_report.schema.json — packaged copy first, repo root from source."""
    res = resources.files("litert_compat").joinpath("schemas/lint_report.schema.json")
    if res.is_file():
        return json.loads(res.read_text(encoding="utf-8"))
    repo_copy = Path(__file__).resolve().parents[3] / "schemas" / "lint_report.schema.json"
    return json.loads(repo_copy.read_text(encoding="utf-8"))


def _probe_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["op"],
        ",".join(record["dtypes"]),
        json.dumps(record["shape_meta"], sort_keys=True),
    )


def build_report(
    classified: list[ClassifiedNode],
    matrix: Matrix,
    model_path: str,
    model_bytes: bytes,
    subgraph_count: int,
    matrix_path: str,
) -> dict[str, Any]:
    partitions, boundaries = simulate_partitions(classified)
    ranked = rank_blocking_ops(classified)

    status_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    for c in classified:
        status_counts[c.verdict.status] = status_counts.get(c.verdict.status, 0) + 1
        entry = c.verdict.matched_entry
        key = entry["provenance"] if entry else "unmatched"
        provenance_counts[key] = provenance_counts.get(key, 0) + 1

    findings = []
    for c in classified:
        entry = c.verdict.matched_entry
        findings.append(
            {
                "subgraph": c.subgraph,
                "node_index": c.node.index,
                "op": c.node.op,
                "custom_code": c.node.custom_code,
                "dtypes": c.dtypes,
                "shape_meta": c.shape_meta,
                "status": c.verdict.status,
                "claimed": c.claimed,
                "reason": c.verdict.reason,
                "provenance": entry["provenance"] if entry else None,
                "conditions": entry.get("conditions") if entry else None,
            }
        )

    rewrite_suggestions = []
    for c in classified:
        entry = c.verdict.matched_entry
        if entry and entry.get("rewrite_hints"):
            rewrite_suggestions.append(
                {
                    "subgraph": c.subgraph,
                    "node_index": c.node.index,
                    "op": c.node.op,
                    "hints": entry["rewrite_hints"],  # verbatim, never generated
                }
            )

    seen_probe_keys = set()
    needs_probe = []
    for c in classified:
        if c.verdict.status != "unknown" or c.is_custom:
            continue
        if c.node.op.startswith("UNKNOWN_BUILTIN_"):
            continue  # unnamed op code: a probe cannot be constructed
        record = {
            "backend": matrix.backend,
            "op": c.node.op,
            "dtypes": c.dtypes,
            "shape_meta": c.shape_meta,
        }
        key = _probe_key(record)
        if key not in seen_probe_keys:
            seen_probe_keys.add(key)
            needs_probe.append(record)
    needs_probe.sort(key=_probe_key)

    custom_ops = [
        {"subgraph": c.subgraph, "node_index": c.node.index, "custom_code": c.node.custom_code}
        for c in classified
        if c.is_custom
    ]

    total = len(classified)
    claimed = sum(1 for c in classified if c.claimed)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": matrix.backend,
        "litert_version": matrix.litert_version,
        "model": {
            "path": model_path,
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "subgraph_count": subgraph_count,
        },
        "matrix": {
            "file": matrix_path,
            "litert_version": matrix.litert_version,
            "generated_at": matrix.doc["generated_at"],
        },
        "summary": {
            "total_ops": total,
            "claimed_ops": claimed,
            "coverage_ops_pct": round(100.0 * claimed / total, 1) if total else 0.0,
            "partition_count": len(partitions),
            "sync_boundaries": boundaries,
            "blocking_op_count": len(ranked),
            "status_counts": status_counts,
            "matched_provenance_counts": provenance_counts,
        },
        "findings": findings,
        "partitions": [
            {
                "subgraph": p.subgraph,
                "index": p.index,
                "first_node": p.first_node,
                "last_node": p.last_node,
                "op_count": p.op_count,
                "ops": list(p.ops),
            }
            for p in partitions
        ],
        "blocking_ops": [
            {
                "subgraph": r.classified.subgraph,
                "node_index": r.classified.node.index,
                "op": r.classified.node.op,
                "status": r.classified.verdict.status,
                "impact_score": r.impact_score,
            }
            for r in ranked
        ],
        "impact_note": IMPACT_NOTE,
        "rewrite_suggestions": rewrite_suggestions,
        "needs_probe": needs_probe,
        "custom_ops": custom_ops,
    }


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _verdict_line(report: dict[str, Any]) -> str:
    s = report["summary"]
    return (
        f"{report['backend']} @ litert {report['litert_version']} — "
        f"{s['coverage_ops_pct']}% of {_plural(s['total_ops'], 'op')} delegated · "
        f"{_plural(s['partition_count'], 'partition')} · "
        f"{_plural(s['blocking_op_count'], 'blocking op')}"
    )


def _meta(shape_meta: dict[str, Any]) -> str:
    return json.dumps(shape_meta, sort_keys=True, separators=(",", ":"))


def _finding_rows(report: dict[str, Any], claimed: bool) -> list[dict[str, Any]]:
    return [f for f in report["findings"] if f["claimed"] is claimed]


def _partition_map_lines(report: dict[str, Any], indent: str) -> list[str]:
    lines = []
    findings_by_key = {(f["subgraph"], f["node_index"]): f for f in report["findings"]}
    for subgraph in sorted({f["subgraph"] for f in report["findings"]}):
        lines.append(f"{indent}subgraph {subgraph}:")
        rows = [f for f in report["findings"] if f["subgraph"] == subgraph]
        partitions = [p for p in report["partitions"] if p["subgraph"] == subgraph]
        by_first = {p["first_node"]: p for p in partitions}
        previous_claimed: bool | None = None
        i = 0
        while i < len(rows):
            f = rows[i]
            if previous_claimed is not None and previous_claimed != f["claimed"]:
                lines.append(f"{indent}  ----- host<->device sync -----")
            if f["claimed"]:
                p = by_first[f["node_index"]]
                ops = ", ".join(p["ops"])
                noun = "op" if p["op_count"] == 1 else "ops"
                lines.append(
                    f"{indent}  [device #{p['index']}] nodes "
                    f"{p['first_node']}-{p['last_node']} ({p['op_count']} {noun}: {ops})"
                )
                i += p["op_count"]
            else:
                detail = findings_by_key[(subgraph, f["node_index"])]
                lines.append(
                    f"{indent}  (host)      node {f['node_index']} "
                    f"{f['op']} ({detail['status']})"
                )
                i += 1
            previous_claimed = f["claimed"]
    return lines


def _resolved_hint_ids(report: dict[str, Any], resolved_rule_ids: frozenset[str]) -> list[str]:
    """transform_ids that appear in the report's hints AND resolve to a rule."""
    return sorted({
        hint["transform_id"]
        for suggestion in report["rewrite_suggestions"]
        for hint in suggestion["hints"]
        if hint.get("transform_id") in resolved_rule_ids
    })


def _fix_command_lines(
    report: dict[str, Any], rules_dir: str, resolved_rule_ids: frozenset[str]
) -> list[str]:
    """The exact edge-fix invocation, when any hint resolves. A container
    lint (`::TFLiteModel@…` path marker) gets a note instead — edge-fix
    rewrites `.tflite` files, not bundles."""
    if not _resolved_hint_ids(report, resolved_rule_ids):
        return []
    if "::" in report["model"]["path"]:
        return ["bundle lint: extract the embedded .tflite to run edge-fix"]
    return [
        f"run: edge-fix {report['model']['path']} "
        f"--matrix {report['matrix']['file']} --rules {rules_dir}"
    ]


def _hint_fixable_line(
    hint: dict[str, Any], rules_dir: str | None, resolved_rule_ids: frozenset[str]
) -> str | None:
    if rules_dir is None or not hint.get("transform_id"):
        return None
    transform_id = hint["transform_id"]
    if transform_id in resolved_rule_ids:
        return f"fixable by edge-fix rule '{transform_id}'"
    return f"transform_id '{transform_id}' does not resolve in {rules_dir}"


def render_text(
    report: dict[str, Any],
    rules_dir: str | None = None,
    resolved_rule_ids: frozenset[str] = frozenset(),
) -> str:
    lines = [_verdict_line(report)]
    lines.append(
        f"model:  {report['model']['path']} "
        f"(sha256 {report['model']['sha256'][:12]}…, "
        f"{report['model']['subgraph_count']} subgraph(s))"
    )
    lines.append(
        f"matrix: {report['matrix']['file']} (generated {report['matrix']['generated_at']})"
    )

    unclaimed = _finding_rows(report, claimed=False)
    if unclaimed:
        lines.append("")
        lines.append("Non-delegated ops:")
        for f in unclaimed:
            lines.append(
                f"  s{f['subgraph']}/n{f['node_index']}  {f['op']}  "
                f"dtypes={','.join(f['dtypes']) or '-'}  {f['status']}  "
                f"provenance={f['provenance'] or '-'}"
            )
            lines.append(f"      reason: {f['reason']}")
            if f["conditions"]:
                lines.append(f"      condition: {f['conditions']}")

    warnings = [f for f in report["findings"] if f["status"] == "incorrect"]
    if warnings:
        lines.append("")
        lines.append("Correctness warnings (delegated but numerics outside tolerance):")
        for f in warnings:
            lines.append(
                f"  s{f['subgraph']}/n{f['node_index']}  {f['op']}  "
                f"dtypes={','.join(f['dtypes']) or '-'}"
                + (f" — {f['conditions']}" if f["conditions"] else "")
            )

    lines.append("")
    lines.append("Partition map (each sync is a host<->device transfer):")
    lines.extend(_partition_map_lines(report, indent="  "))

    if report["blocking_ops"]:
        lines.append("")
        lines.append(f"Blocking ops, ranked ({report['impact_note']}):")
        for rank, b in enumerate(report["blocking_ops"], start=1):
            lines.append(
                f"  {rank}. s{b['subgraph']}/n{b['node_index']} {b['op']} "
                f"({b['status']}) — impact {b['impact_score']}"
            )

    if report["rewrite_suggestions"]:
        lines.append("")
        lines.append("Rewrite suggestions (verbatim from the matrix, never generated):")
        for s in report["rewrite_suggestions"]:
            lines.append(f"  s{s['subgraph']}/n{s['node_index']} {s['op']}:")
            for hint in s["hints"]:
                lines.append(f"    symptom: {hint['symptom']}")
                lines.append(f"    rewrite: {hint['rewrite']}")
                if hint.get("expected_effect"):
                    lines.append(f"    expected effect: {hint['expected_effect']}")
                fixable = _hint_fixable_line(hint, rules_dir, resolved_rule_ids)
                if fixable is not None:
                    lines.append(f"    {fixable}")
        if rules_dir is not None:
            lines += [
                f"  {line}"
                for line in _fix_command_lines(report, rules_dir, resolved_rule_ids)
            ]

    if report["needs_probe"]:
        lines.append("")
        lines.append("Needs probe (complete lookup signatures; no measured entry matched):")
        for p in report["needs_probe"]:
            lines.append(
                f"  {p['op']} dtypes={','.join(p['dtypes']) or '-'} "
                f"shape_meta={_meta(p['shape_meta'])} backend={p['backend']}"
            )

    if report["custom_ops"]:
        lines.append("")
        lines.append("Custom ops (outside the matrix vocabulary; not probeable):")
        for c in report["custom_ops"]:
            lines.append(f"  s{c['subgraph']}/n{c['node_index']} CUSTOM {c['custom_code']!r}")

    return "\n".join(lines) + "\n"


def render_markdown(
    report: dict[str, Any],
    rules_dir: str | None = None,
    resolved_rule_ids: frozenset[str] = frozenset(),
) -> str:
    s = report["summary"]
    lines = [
        f"# edge-lint report — `{Path(report['model']['path']).name}`",
        "",
        f"**{_verdict_line(report)}**",
        "",
        f"- model: `{report['model']['path']}` "
        f"(sha256 `{report['model']['sha256']}`, "
        f"{report['model']['subgraph_count']} subgraph(s))",
        f"- matrix: `{report['matrix']['file']}` "
        f"(generated {report['matrix']['generated_at']})",
        f"- sync boundaries: {s['sync_boundaries']}",
        "",
        "## Per-op verdicts",
        "",
        "| loc | op | dtypes | status | claimed | provenance | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in report["findings"]:
        lines.append(
            f"| s{f['subgraph']}/n{f['node_index']} | {f['op']} "
            f"| {','.join(f['dtypes']) or '-'} | {f['status']} "
            f"| {'yes' if f['claimed'] else 'no'} | {f['provenance'] or '-'} "
            f"| {f['reason']} |"
        )

    lines += ["", "## Partition map", "", "```"]
    lines.extend(_partition_map_lines(report, indent=""))
    lines += ["```", ""]

    if report["blocking_ops"]:
        lines += [
            "## Blocking ops (ranked)",
            "",
            f"> {report['impact_note']}",
            "",
            "| rank | loc | op | status | impact |",
            "|---|---|---|---|---|",
        ]
        for rank, b in enumerate(report["blocking_ops"], start=1):
            lines.append(
                f"| {rank} | s{b['subgraph']}/n{b['node_index']} | {b['op']} "
                f"| {b['status']} | {b['impact_score']} |"
            )
        lines.append("")

    if report["rewrite_suggestions"]:
        lines += ["## Rewrite suggestions", "", "_Verbatim from the matrix, never generated._", ""]
        for suggestion in report["rewrite_suggestions"]:
            lines.append(
                f"- **s{suggestion['subgraph']}/n{suggestion['node_index']} "
                f"{suggestion['op']}**"
            )
            for hint in suggestion["hints"]:
                lines.append(f"  - symptom: {hint['symptom']}")
                lines.append(f"  - rewrite: {hint['rewrite']}")
                if hint.get("expected_effect"):
                    lines.append(f"  - expected effect: {hint['expected_effect']}")
                fixable = _hint_fixable_line(hint, rules_dir, resolved_rule_ids)
                if fixable is not None:
                    lines.append(f"  - {fixable}")
        if rules_dir is not None:
            for line in _fix_command_lines(report, rules_dir, resolved_rule_ids):
                lines.append(f"- _{line}_")
        lines.append("")

    if report["needs_probe"]:
        lines += [
            "## Needs probe",
            "",
            "Complete lookup signatures — sufficient to construct a probe without "
            "reopening the model.",
            "",
            "| op | dtypes | shape_meta | backend |",
            "|---|---|---|---|",
        ]
        for p in report["needs_probe"]:
            lines.append(
                f"| {p['op']} | {','.join(p['dtypes']) or '-'} "
                f"| `{_meta(p['shape_meta'])}` | {p['backend']} |"
            )
        lines.append("")

    if report["custom_ops"]:
        lines += ["## Custom ops", ""]
        for c in report["custom_ops"]:
            lines.append(
                f"- s{c['subgraph']}/n{c['node_index']} CUSTOM `{c['custom_code']}` "
                "(outside the matrix vocabulary; not probeable)"
            )
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
