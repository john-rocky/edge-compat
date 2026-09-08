"""`release-check`: carry-forward → probe → re-measure → upgrade → diff → report.

One command per runtime release (spec §8.3). No new schema: probe output is
matrix entries, the report embeds `matrix diff` output, and the exit-code
contract (0 nothing changed / 1 semantic changes / 2 error) lets the owner's
existing release watcher trigger on it.

Upgrade rules (Data Integrity + micrograph honesty, spelled out):
- A probe result may upgrade an `inferred` entry back to `measured`, with
  `evidence.source_model: "probe:<fixture-id>"` pinning the probe scale.
- A probe result that CONTRADICTS an entry whose ancestry is a full-model
  measurement (evidence.source_model not `probe:*`) is a CONFLICT: surfaced in
  the report, entry left `inferred`, never silently overwritten either way.
- A runner returning `unknown` is not a measurement: the entry stays inferred.
- No runner available for a backend → its entries stay `inferred` and are
  listed as remaining work. The `inferred` count IS the staleness debt.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_compat.lint.classify import ClassifiedNode, classify_model
from litert_compat.lint.partition import simulate_partitions
from litert_compat.matrix.canonical import (
    constraints_key,
    load_json,
    signature,
    sort_entries,
    write_canonical,
)
from litert_compat.matrix.carry import carry_forward
from litert_compat.matrix.diff import diff_documents
from litert_compat.matrix.diff import render_text as render_diff_text
from litert_compat.matrix.query import Matrix, MatrixLookupTieError
from litert_compat.matrix.validate import MatrixValidationError, validate_document
from litert_compat.parser.reader import TfliteParseError, parse_tflite_file
from litert_compat.probe.gen import generate_fixture
from litert_compat.probe.runners import Runner, RunnerError, RunnerResult, Tolerance
from litert_compat.probe.signatures import (
    ProbeSignature,
    UnprobeableSignatureError,
    signature_from_entry,
    signatures_from_lint_report,
)


class ReleaseCheckError(ValueError):
    """Usage-level error (exit 2): bad inputs, duplicate backends, ..."""


@dataclass(frozen=True)
class SnapshotInput:
    path: Path
    doc: dict[str, Any]


def _probe_evidence(fixture_id: str, litert_version: str, date: str) -> dict[str, str]:
    return {
        "source_model": f"probe:{fixture_id}",
        "litert_version": litert_version,
        "date": date,
    }


def _full_model_ancestry(entry: dict[str, Any]) -> bool:
    source = (entry.get("evidence") or {}).get("source_model")
    return bool(source) and not str(source).startswith("probe:")


def _record(
    kind: str,
    op: str,
    dtypes: list[str],
    constraints: dict[str, Any] | None,
    *,
    disposition: str,
    fixture: str | None = None,
    shape_meta: dict[str, Any] | None = None,
    status: str | None = None,
    max_abs_diff: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "op": op,
        "dtypes": dtypes,
        "constraints": constraints or {},
        "shape_meta": shape_meta,
        "fixture": fixture,
        "disposition": disposition,
        "status": status,
        "max_abs_diff": max_abs_diff,
        "note": note,
    }


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        record["op"],
        ",".join(record["dtypes"]),
        constraints_key(record["constraints"]),
        record["kind"],
    )


def _run_probe(
    sig: ProbeSignature, runner: Runner, probe_dir: Path
) -> tuple[str | None, RunnerResult | None, str | None]:
    """-> (fixture_id, result, error_note). Generation and runner errors are
    reported, never guessed around."""
    try:
        fixture = generate_fixture(sig, probe_dir)
    except UnprobeableSignatureError as exc:
        return None, None, exc.reason
    try:
        return fixture.fixture_id, runner.run(fixture, sig.backend), None
    except RunnerError as exc:
        return fixture.fixture_id, None, str(exc)


def check_snapshot(
    old_doc: dict[str, Any],
    *,
    to_litert_version: str,
    generated_at: str,
    needs_probe: list[ProbeSignature],
    runner: Runner | None,
    runner_reason: str | None,
    probe_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carry one snapshot forward, probe what can be probed, apply results.

    -> (new_doc, backend_report). `needs_probe` must already be filtered to
    this snapshot's backend.
    """
    backend = old_doc["backend"]
    new_doc = carry_forward(
        old_doc, to_litert_version=to_litert_version, generated_at=generated_at
    )

    existing_signatures = {signature(entry) for entry in new_doc["entries"]}
    records: list[dict[str, Any]] = []

    # 1. Inferred entries of the carried snapshot: re-measure and upgrade.
    for entry in sorted(new_doc["entries"], key=lambda e: signature(e)):
        if entry.get("provenance") != "inferred":
            continue
        dtypes = list(entry.get("dtypes") or [])
        constraints = entry.get("constraints")

        def entry_record(**kwargs: Any) -> dict[str, Any]:
            return _record("entry", entry["op"], dtypes, constraints, **kwargs)  # noqa: B023

        if runner is None:
            records.append(entry_record(disposition="no_runner", note=runner_reason))
            continue
        try:
            sig = signature_from_entry(backend, entry)
        except UnprobeableSignatureError as exc:
            records.append(entry_record(disposition="unprobeable", note=exc.reason))
            continue
        fixture_id, result, error = _run_probe(sig, runner, probe_dir)
        if result is None:
            disposition = "unprobeable" if fixture_id is None else "runner_error"
            records.append(
                entry_record(
                    disposition=disposition,
                    fixture=fixture_id,
                    shape_meta=sig.shape_meta,
                    note=error,
                )
            )
            continue
        if result.status == "unknown":
            records.append(
                entry_record(
                    disposition="unknown_result",
                    fixture=fixture_id,
                    shape_meta=sig.shape_meta,
                    status=result.status,
                    max_abs_diff=result.max_abs_diff,
                    note=result.note,
                )
            )
            continue
        if result.status != entry["status"] and _full_model_ancestry(entry):
            records.append(
                entry_record(
                    disposition="conflict",
                    fixture=fixture_id,
                    shape_meta=sig.shape_meta,
                    status=result.status,
                    max_abs_diff=result.max_abs_diff,
                    note=(
                        f"probe measured {result.status!r} but the entry's "
                        f"{entry['status']!r} comes from a full-model measurement "
                        f"({entry['evidence']['source_model']}); kept inferred — "
                        "re-measure at full-model scale to resolve"
                    ),
                )
            )
            continue
        entry["status"] = result.status
        entry["provenance"] = "measured"
        entry["evidence"] = _probe_evidence(fixture_id or "", to_litert_version, generated_at)
        records.append(
            entry_record(
                disposition="upgraded",
                fixture=fixture_id,
                shape_meta=sig.shape_meta,
                status=result.status,
                max_abs_diff=result.max_abs_diff,
                note=result.note,
            )
        )

    # 2. Accumulated needs-probe signatures: measure and create new entries.
    for sig in sorted(set(needs_probe), key=ProbeSignature.sort_key):
        meta = sig.shape_meta
        constraints: dict[str, Any] = {"dynamic_shape": bool(meta.get("dynamic_shape", False))}
        if isinstance(meta.get("rank"), int):
            constraints["rank"] = meta["rank"]
        dtypes = list(sig.dtypes)

        def sig_record(**kwargs: Any) -> dict[str, Any]:
            return _record(
                "signature", sig.op, dtypes, constraints, shape_meta=meta, **kwargs  # noqa: B023
            )

        new_entry = {
            "op": sig.op,
            "constraints": constraints,
            "status": "",
            "provenance": "measured",
        }
        if dtypes:
            new_entry["dtypes"] = dtypes
        if signature(new_entry) in existing_signatures:
            records.append(
                sig_record(disposition="duplicate", note="already covered by an entry")
            )
            continue
        if runner is None:
            records.append(sig_record(disposition="no_runner", note=runner_reason))
            continue
        fixture_id, result, error = _run_probe(sig, runner, probe_dir)
        if result is None:
            disposition = "unprobeable" if fixture_id is None else "runner_error"
            records.append(sig_record(disposition=disposition, fixture=fixture_id, note=error))
            continue
        if result.status == "unknown":
            records.append(
                sig_record(
                    disposition="unknown_result",
                    fixture=fixture_id,
                    status=result.status,
                    max_abs_diff=result.max_abs_diff,
                    note=result.note,
                )
            )
            continue
        new_entry["status"] = result.status
        new_entry["evidence"] = _probe_evidence(
            fixture_id or "", to_litert_version, generated_at
        )
        new_doc["entries"].append(new_entry)
        existing_signatures.add(signature(new_entry))
        records.append(
            sig_record(
                disposition="created",
                fixture=fixture_id,
                status=result.status,
                max_abs_diff=result.max_abs_diff,
                note=result.note,
            )
        )

    sort_entries(new_doc)
    findings = validate_document(new_doc)
    if findings:
        raise MatrixValidationError(f"release-check output for {backend}", findings)

    records.sort(key=_record_sort_key)
    dispositions = sorted({r["disposition"] for r in records})
    counts = {d: sum(1 for r in records if r["disposition"] == d) for d in dispositions}
    remaining_inferred = sum(
        1 for entry in new_doc["entries"] if entry.get("provenance") == "inferred"
    )
    backend_report = {
        "backend": backend,
        "old": {
            "litert_version": old_doc["litert_version"],
            "generated_at": old_doc["generated_at"],
        },
        "new": {"litert_version": to_litert_version, "generated_at": generated_at},
        "runner": runner.name if runner else None,
        "runner_reason": runner_reason,
        "probes": records,
        "counts": counts,
        "remaining_inferred": remaining_inferred,
        "diff": diff_documents(old_doc, new_doc),
    }
    return new_doc, backend_report


def _lint_summary(classified: list[ClassifiedNode]) -> dict[str, Any]:
    total = len(classified)
    claimed = sum(1 for c in classified if c.claimed)
    partitions, _boundaries = simulate_partitions(classified)
    return {
        "total_ops": total,
        "claimed_ops": claimed,
        "coverage_ops_pct": round(100.0 * claimed / total, 1) if total else 0.0,
        "partition_count": len(partitions),
    }


def manifest_relint(
    manifest_path: Path, pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-lint every manifest model against old and new snapshots.

    -> (affected, errors). A model appears in `affected` only when at least
    one node's verdict differs between the snapshots of some backend.
    Failures are collected, never fail-fast (the Phase 3 batch convention).
    """
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "model" not in rows[0]:
        raise ReleaseCheckError(f"{manifest_path}: manifest has no 'model' column")

    models = sorted({row["model"].strip() for row in rows if (row.get("model") or "").strip()})
    affected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for model_rel in models:
        model_path = manifest_path.parent / model_rel
        try:
            parsed = parse_tflite_file(model_path)
        except TfliteParseError as exc:
            errors.append({"model": model_rel, "error": f"TfliteParseError: {exc}"})
            continue
        except OSError as exc:
            # strerror only: no absolute paths, so reports stay deterministic
            # across machines.
            detail = exc.strerror or "unreadable"
            errors.append({"model": model_rel, "error": f"{type(exc).__name__}: {detail}"})
            continue
        backends_changed = []
        for backend in sorted(pairs):
            old_doc, new_doc = pairs[backend]
            try:
                old_classified = classify_model(parsed, Matrix(old_doc))
                new_classified = classify_model(parsed, Matrix(new_doc))
            except MatrixLookupTieError as exc:
                errors.append({"model": model_rel, "error": f"MatrixLookupTieError: {exc}"})
                break
            changed_ops = [
                {
                    "subgraph": old.subgraph,
                    "node_index": old.node.index,
                    "op": old.node.op,
                    "from": old.verdict.status,
                    "to": new.verdict.status,
                }
                for old, new in zip(old_classified, new_classified, strict=True)
                if old.verdict.status != new.verdict.status
            ]
            if changed_ops:
                backends_changed.append(
                    {
                        "backend": backend,
                        "old": _lint_summary(old_classified),
                        "new": _lint_summary(new_classified),
                        "changed_ops": changed_ops,
                    }
                )
        if backends_changed:
            affected.append({"model": model_rel, "backends": backends_changed})
    return affected, errors


def release_check(
    snapshots: list[Path],
    *,
    to_litert_version: str,
    generated_at: str,
    lint_report_docs: list[dict[str, Any]],
    runners: list[Runner],
    out_dir: Path,
    tolerance: Tolerance,
    manifest: Path | None = None,
    probe_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline; writes new snapshots under `out_dir` and
    returns the release report object (deterministic)."""
    inputs: list[SnapshotInput] = []
    seen_backends: set[str] = set()
    for path in snapshots:
        doc = load_json(path)
        findings = validate_document(doc)
        if findings:
            raise MatrixValidationError(str(path), findings)
        backend = doc["backend"]
        if backend in seen_backends:
            raise ReleaseCheckError(
                f"two input snapshots share backend {backend!r}; "
                "release-check takes one snapshot per backend"
            )
        if doc["litert_version"] == to_litert_version:
            raise ReleaseCheckError(
                f"{path}: snapshot is already at litert_version {to_litert_version}"
            )
        seen_backends.add(backend)
        inputs.append(SnapshotInput(path=path, doc=doc))

    needs_probe: list[ProbeSignature] = []
    for doc in lint_report_docs:
        needs_probe.extend(signatures_from_lint_report(doc))
    unmatched_backends = sorted(
        {sig.backend for sig in needs_probe if sig.backend not in seen_backends}
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = probe_dir or out_dir / "probes"

    availability: dict[str, str | None] = {r.name: r.availability() for r in runners}
    backend_reports = []
    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for snapshot in sorted(inputs, key=lambda s: s.doc["backend"]):
        backend = snapshot.doc["backend"]
        runner: Runner | None = None
        runner_reason: str | None = None
        claiming = [r for r in runners if r.claims(backend)]
        available = [r for r in claiming if availability[r.name] is None]
        if available:
            runner = available[0]
        elif claiming:
            runner_reason = "; ".join(
                f"{r.name}: {availability[r.name]}" for r in claiming
            )
        else:
            runner_reason = f"no runner claims backend {backend!r}"

        new_doc, backend_report = check_snapshot(
            snapshot.doc,
            to_litert_version=to_litert_version,
            generated_at=generated_at,
            needs_probe=[sig for sig in needs_probe if sig.backend == backend],
            runner=runner,
            runner_reason=runner_reason,
            probe_dir=probe_dir,
        )
        out_path = out_dir / f"{backend}__{to_litert_version}.json"
        write_canonical(new_doc, out_path)
        backend_report["snapshot"] = {"old": snapshot.path.name, "new": out_path.name}
        backend_reports.append(backend_report)
        pairs[backend] = (snapshot.doc, new_doc)

    models: dict[str, Any] = {"manifest": None, "affected": [], "errors": []}
    if manifest is not None:
        affected, errors = manifest_relint(manifest, pairs)
        models = {"manifest": manifest.name, "affected": affected, "errors": errors}

    return {
        "to_litert_version": to_litert_version,
        "generated_at": generated_at,
        "tolerance": {"absolute": tolerance.absolute, "relative": tolerance.relative},
        "runners": [
            {"name": r.name, "available": availability[r.name] is None,
             "reason": availability[r.name]}
            for r in sorted(runners, key=lambda r: r.name)
        ],
        "backends": backend_reports,
        "needs_probe_unmatched_backends": unmatched_backends,
        "models": models,
        "has_changes": any(b["diff"]["has_changes"] for b in backend_reports),
    }


def _fmt_diff_value(value: float | None) -> str:
    return f"{value:.3e}" if value is not None else "-"


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# edge-compat release check — {report['to_litert_version']} "
        f"({report['generated_at']})",
        "",
        "Semantic changes: **" + ("yes" if report["has_changes"] else "no") + "**. "
        f"Numeric tolerance vs CPU reference: abs {report['tolerance']['absolute']:g}, "
        f"rel {report['tolerance']['relative']:g}.",
        "",
        "## Runners",
        "",
    ]
    for runner in report["runners"]:
        status = "available" if runner["available"] else f"unavailable — {runner['reason']}"
        lines.append(f"- `{runner['name']}`: {status}")

    for backend in report["backends"]:
        lines += [
            "",
            f"## {backend['backend']}: {backend['old']['litert_version']} → "
            f"{backend['new']['litert_version']}",
            "",
            f"- snapshot: `{backend['snapshot']['old']}` → `{backend['snapshot']['new']}`",
            "- runner: "
            + (f"`{backend['runner']}`" if backend["runner"]
               else f"none — {backend['runner_reason']}"),
            f"- remaining `inferred` entries (staleness debt): "
            f"{backend['remaining_inferred']}",
        ]
        if backend["counts"]:
            counted = ", ".join(f"{k}={v}" for k, v in sorted(backend["counts"].items()))
            lines.append(f"- probe outcomes: {counted}")
        if backend["probes"]:
            lines += [
                "",
                "| op | dtypes | constraints | kind | disposition | status "
                "| max_abs_diff | note |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for record in backend["probes"]:
                lines.append(
                    f"| {record['op']} | {','.join(record['dtypes']) or '-'} "
                    f"| `{constraints_key(record['constraints'])}` | {record['kind']} "
                    f"| {record['disposition']} | {record['status'] or '-'} "
                    f"| {_fmt_diff_value(record['max_abs_diff'])} "
                    f"| {record['note'] or '-'} |"
                )
        conflicts = [r for r in backend["probes"] if r["disposition"] == "conflict"]
        if conflicts:
            lines += ["", "**Conflicts (probe vs full-model measurement; unresolved):**", ""]
            for record in conflicts:
                lines.append(f"- {record['op']}: {record['note']}")
        lines += ["", "```", render_diff_text(backend["diff"]), "```"]

    if report["needs_probe_unmatched_backends"]:
        lines += [
            "",
            "## Needs-probe signatures without a snapshot",
            "",
            "These backends appear in lint `needs_probe` output but no input "
            "snapshot covers them: "
            + ", ".join(f"`{b}`" for b in report["needs_probe_unmatched_backends"]),
        ]

    models = report["models"]
    if models["manifest"]:
        lines += ["", f"## Models re-linted (`{models['manifest']}`)", ""]
        if not models["affected"]:
            lines.append("No model's lint verdicts changed.")
        for model in models["affected"]:
            lines.append(f"### `{model['model']}`")
            lines.append("")
            for backend in model["backends"]:
                old, new = backend["old"], backend["new"]
                lines.append(
                    f"- **{backend['backend']}**: coverage "
                    f"{old['coverage_ops_pct']}% → {new['coverage_ops_pct']}%, "
                    f"partitions {old['partition_count']} → {new['partition_count']}"
                )
                for change in backend["changed_ops"]:
                    lines.append(
                        f"  - s{change['subgraph']}/n{change['node_index']} "
                        f"{change['op']}: {change['from']} → {change['to']}"
                    )
            lines.append("")
        if models["errors"]:
            lines += ["**Manifest errors (collected, not fail-fast):**", ""]
            for error in models["errors"]:
                lines.append(f"- `{error['model']}`: {error['error']}")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
