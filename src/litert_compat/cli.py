"""edge-compat command line interface.

Exit-code contract (stable, consumed by CI and the owner's release watcher):

- `matrix validate`:       0 pass · 1 findings · 2 usage error
- `matrix import`:         0 ok · 1 CSV data errors / invalid result · 2 usage error
- `matrix diff`:           0 no semantic difference · 1 differences · 2 usage error
                           (invalid inputs and backend mismatch are usage errors,
                           so exit 1 always means real semantic change)
- `matrix carry-forward`:  0 ok · 2 usage error (incl. invalid input snapshot)
- `probe gen`:             0 all generated · 1 unprobeable signatures · 2 usage error
- `release-check`:         0 nothing changed · 1 semantic changes found · 2 error
                           (headless-safe; unavailable runners are remaining work,
                           not errors)
- `release-gate`:          0 pass · 1 findings · 2 usage error (unreadable scope
                           file, invalid cards, missing site dist)
- `freshness …`:           see litert_compat.freshness.cli (Phase 11)
- `device-run …`:          see litert_compat.device_runs.cli (Phase 13)
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from litert_compat.device_runs.cli import device_run_app
from litert_compat.freshness.cli import freshness_app
from litert_compat.matrix.canonical import canonical_dumps, load_json, write_canonical
from litert_compat.matrix.carry import carry_forward as carry_forward_doc
from litert_compat.matrix.diff import BackendMismatchError, diff_documents, render_text
from litert_compat.matrix.importer import CsvImportError, import_csv
from litert_compat.matrix.validate import (
    MatrixValidationError,
    provenance_counts,
    validate_document,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Delegate compatibility toolkit for LiteRT.",
)
matrix_app = typer.Typer(no_args_is_help=True, help="Compatibility matrix tools.")
app.add_typer(matrix_app, name="matrix")
probe_app = typer.Typer(
    no_args_is_help=True,
    help="Probe fixtures: minimal single-op models for re-measurement (Phase 8).",
)
app.add_typer(probe_app, name="probe")
app.add_typer(freshness_app, name="freshness")
app.add_typer(device_run_app, name="device-run")

MatrixFileArg = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Matrix JSON snapshot.")
]


def _load_json_or_exit(path: Path) -> Any:
    try:
        return load_json(path)
    except json.JSONDecodeError as exc:
        typer.echo(f"error: {path}: not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc


def _load_valid_or_exit(path: Path) -> dict[str, Any]:
    doc = _load_json_or_exit(path)
    findings = validate_document(doc)
    if findings:
        typer.echo(f"error: {path} is not a valid matrix snapshot:", err=True)
        for finding in findings:
            typer.echo(f"  [{finding.code}] {finding.message}", err=True)
        raise typer.Exit(2)
    return doc


def _echo_counts(doc: dict[str, Any]) -> None:
    counts = provenance_counts(doc)
    rendered = " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    typer.echo(f"  provenance: {rendered}")


@matrix_app.command()
def validate(file: MatrixFileArg) -> None:
    """Validate a snapshot: JSON Schema plus semantic lint, with provenance counts."""
    doc = _load_json_or_exit(file)
    findings = validate_document(doc)
    if findings:
        typer.echo(f"{file}: FAIL — {len(findings)} finding(s)")
        for finding in findings:
            typer.echo(f"  [{finding.code}] {finding.message}")
        raise typer.Exit(1)
    typer.echo(f"{file}: PASS")
    typer.echo(
        f"  backend={doc['backend']} litert_version={doc['litert_version']} "
        f"entries={len(doc['entries'])}"
    )
    _echo_counts(doc)


@matrix_app.command("import")
def import_cmd(
    csv_file: Annotated[
        Path, typer.Option("--csv", exists=True, dir_okay=False, help="Input CSV table.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output matrix JSON path.")],
    litert_version: Annotated[str, typer.Option("--litert-version")],
    backend: Annotated[str, typer.Option("--backend")],
    device: Annotated[str | None, typer.Option(help="target.device")] = None,
    soc: Annotated[str | None, typer.Option(help="target.soc")] = None,
    driver: Annotated[str | None, typer.Option(help="target.driver")] = None,
    generated_at: Annotated[
        str | None,
        typer.Option(help="YYYY-MM-DD; defaults to today. Pass it for reproducible output."),
    ] = None,
) -> None:
    """Import the owner's CSV op table into a valid matrix snapshot."""
    target = {"device": device, "soc": soc, "driver": driver}
    try:
        result = import_csv(
            csv_file,
            litert_version=litert_version,
            backend=backend,
            generated_at=generated_at or datetime.date.today().isoformat(),
            target={k: v for k, v in target.items() if v} or None,
        )
    except CsvImportError as exc:
        typer.echo("import FAILED:", err=True)
        for error in exc.errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(1) from exc

    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)

    findings = validate_document(result.doc)
    if findings:
        typer.echo("import produced an invalid snapshot (nothing written):", err=True)
        for finding in findings:
            typer.echo(f"  [{finding.code}] {finding.message}", err=True)
        raise typer.Exit(1)

    write_canonical(result.doc, out)
    typer.echo(f"{out}: wrote {len(result.doc['entries'])} entries")
    _echo_counts(result.doc)


@matrix_app.command()
def diff(
    old: MatrixFileArg,
    new: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Semantic change summary between two snapshots of the same backend."""
    old_doc = _load_valid_or_exit(old)
    new_doc = _load_valid_or_exit(new)
    try:
        result = diff_documents(old_doc, new_doc)
    except BackendMismatchError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    if json_output:
        typer.echo(canonical_dumps(result), nl=False)
    else:
        typer.echo(render_text(result))
    raise typer.Exit(1 if result["has_changes"] else 0)


@matrix_app.command("carry-forward")
def carry_forward_cmd(
    file: MatrixFileArg,
    to_litert_version: Annotated[str, typer.Option("--to-litert-version")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output path for the new snapshot.")],
    generated_at: Annotated[
        str | None,
        typer.Option(help="YYYY-MM-DD; defaults to today. Pass it for reproducible output."),
    ] = None,
) -> None:
    """Produce the next-version snapshot; measured entries downgrade to inferred."""
    if out.resolve() == file.resolve():
        typer.echo("error: carry-forward never edits in place; choose a new --out path", err=True)
        raise typer.Exit(2)
    doc = _load_valid_or_exit(file)
    try:
        new_doc = carry_forward_doc(
            doc,
            to_litert_version=to_litert_version,
            generated_at=generated_at or datetime.date.today().isoformat(),
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    write_canonical(new_doc, out)
    downgraded = sum(1 for entry in doc["entries"] if entry.get("provenance") == "measured")
    typer.echo(
        f"{out}: carried {doc['litert_version']} → {to_litert_version}; "
        f"{downgraded} measured entries downgraded to inferred"
    )
    _echo_counts(new_doc)


@probe_app.command("gen")
def probe_gen(
    from_lint: Annotated[
        list[Path] | None,
        typer.Option(
            "--from-lint", exists=True, dir_okay=False, readable=True,
            help="edge-lint --json report; its needs_probe signatures become probes.",
        ),
    ] = None,
    from_matrix: Annotated[
        list[Path] | None,
        typer.Option(
            "--from-matrix", exists=True, dir_okay=False, readable=True,
            help="Matrix snapshot; its inferred entries become probes.",
        ),
    ] = None,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", "-o", help="Fixture output directory.")
    ] = Path("data/probes"),
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Generate probe fixtures from lookup signatures. Deterministic:
    signature → stable fixture id/path, byte-identical on regeneration."""
    from litert_compat.probe.gen import generate_fixture
    from litert_compat.probe.signatures import (
        ProbeSignature,
        UnprobeableSignatureError,
        signature_from_entry,
        signatures_from_lint_report,
    )

    if not from_lint and not from_matrix:
        typer.echo("error: pass at least one --from-lint or --from-matrix input", err=True)
        raise typer.Exit(2)

    signatures: set[ProbeSignature] = set()
    unprobeable_entries: list[tuple[str, str, str]] = []  # (backend, op, reason)
    for path in from_lint or []:
        doc = _load_json_or_exit(path)
        try:
            signatures.update(signatures_from_lint_report(doc))
        except (ValueError, KeyError, TypeError) as exc:
            typer.echo(f"error: {path}: not a usable lint report: {exc}", err=True)
            raise typer.Exit(2) from exc
    for path in from_matrix or []:
        doc = _load_valid_or_exit(path)
        for entry in doc["entries"]:
            if entry.get("provenance") != "inferred":
                continue
            try:
                signatures.add(signature_from_entry(doc["backend"], entry))
            except UnprobeableSignatureError as exc:
                unprobeable_entries.append((doc["backend"], entry["op"], exc.reason))

    rows: list[dict[str, Any]] = []
    for sig in sorted(signatures, key=ProbeSignature.sort_key):
        row = sig.as_dict()
        try:
            fixture = generate_fixture(sig, out_dir)
            row.update(status="generated", fixture=fixture.fixture_id,
                       path=str(fixture.path), reason=None)
        except UnprobeableSignatureError as exc:
            row.update(status="unprobeable", fixture=None, path=None, reason=exc.reason)
        rows.append(row)
    for backend, op, reason in sorted(unprobeable_entries):
        rows.append({
            "backend": backend, "op": op, "dtypes": None, "shape_meta": None,
            "status": "unprobeable", "fixture": None, "path": None, "reason": reason,
        })

    if json_output:
        typer.echo(canonical_dumps(rows), nl=False)
    else:
        for row in rows:
            if row["status"] == "generated":
                typer.echo(f"generated   {row['fixture']}  ->  {row['path']}")
            else:
                typer.echo(
                    f"unprobeable {row['op']} ({row['backend']}): {row['reason']}"
                )
    raise typer.Exit(1 if any(r["status"] == "unprobeable" for r in rows) else 0)


@app.command("release-check")
def release_check_cmd(
    snapshots: Annotated[
        list[Path],
        typer.Argument(
            exists=True, dir_okay=False, readable=True,
            help="Matrix snapshots to carry forward, one per backend.",
        ),
    ],
    to_litert_version: Annotated[str, typer.Option("--to-litert-version")],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", "-o",
                           help="Output directory: new snapshots, probes/, release report.")
    ],
    lint_report: Annotated[
        list[Path] | None,
        typer.Option(
            "--lint-report", exists=True, dir_okay=False, readable=True,
            help="Accumulated edge-lint --json reports; their needs_probe "
                 "signatures are probed too.",
        ),
    ] = None,
    runner: Annotated[
        list[str] | None,
        typer.Option(
            "--runner",
            help="Runners to use: cpu | webgpu_mac | gpu_mldrift_adb. Default: all "
                 "(unavailable ones are reported as remaining work).",
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, readable=True,
            help="Phase 3 build-all manifest: re-lint its models and report "
                 "changed verdicts.",
        ),
    ] = None,
    generated_at: Annotated[
        str | None,
        typer.Option(help="YYYY-MM-DD; defaults to today. Pass it for reproducible output."),
    ] = None,
    tolerance_abs: Annotated[
        float, typer.Option("--tolerance-abs", help="Absolute numeric tolerance vs CPU.")
    ] = 1e-5,
    tolerance_rel: Annotated[
        float, typer.Option("--tolerance-rel", help="Relative numeric tolerance vs CPU.")
    ] = 1e-3,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the JSON report instead of markdown.")
    ] = False,
) -> None:
    """One command per runtime release: carry-forward → probe → re-measure →
    upgrade → diff → report. Headless-safe: unavailable backends stay
    `inferred` and are listed as remaining work."""
    from litert_compat.probe.release import (
        ReleaseCheckError,
        release_check,
        render_report_markdown,
    )
    from litert_compat.probe.runners import Tolerance, runner_registry

    tolerance = Tolerance(absolute=tolerance_abs, relative=tolerance_rel)
    registry = runner_registry(tolerance)
    names = runner or list(registry)
    unknown = sorted(set(names) - set(registry))
    if unknown:
        typer.echo(
            f"error: unknown runner(s) {', '.join(unknown)} "
            f"(known: {', '.join(sorted(registry))})", err=True,
        )
        raise typer.Exit(2)

    lint_docs = [_load_json_or_exit(path) for path in lint_report or []]
    try:
        report = release_check(
            snapshots,
            to_litert_version=to_litert_version,
            generated_at=generated_at or datetime.date.today().isoformat(),
            lint_report_docs=lint_docs,
            runners=[registry[name] for name in dict.fromkeys(names)],
            out_dir=out_dir,
            tolerance=tolerance,
            manifest=manifest,
        )
    except (ReleaseCheckError, MatrixValidationError, ValueError, KeyError, TypeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    markdown = render_report_markdown(report)
    (out_dir / "release_report.json").write_text(canonical_dumps(report), encoding="utf-8")
    (out_dir / "release_report.md").write_text(markdown, encoding="utf-8")
    typer.echo(canonical_dumps(report) if json_output else markdown, nl=False)
    raise typer.Exit(1 if report["has_changes"] else 0)


@app.command("release-gate")
def release_gate_cmd(
    repo_root: Annotated[
        Path,
        typer.Option(
            "--repo-root", exists=True, file_okay=False,
            help="Repository root holding README.md, cards/, llms.txt.",
        ),
    ] = Path("."),
    scope: Annotated[
        Path | None,
        typer.Option(
            "--scope",
            help="Approved public benchmark scope (owner-editable data). "
                 "Default: <repo-root>/data/release_scope.json.",
        ),
    ] = None,
    site_dist: Annotated[
        Path | None,
        typer.Option(
            "--site-dist", exists=True, file_okay=False,
            help="Built site output (site/dist); every HTML page is checked too.",
        ),
    ] = None,
) -> None:
    """Pre-publication gate (Phase 12): benchmark-scope guard, disclosure-line
    presence, example-provenance banners. Deterministic; run by RELEASE.md and
    CI before anything goes public."""
    from litert_compat.release_gate import ReleaseGateUsageError, run_gate

    scope_path = scope if scope is not None else repo_root / "data" / "release_scope.json"
    try:
        findings = run_gate(repo_root, scope_path, site_dist)
    except ReleaseGateUsageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    if findings:
        typer.echo(f"release-gate: FAIL — {len(findings)} finding(s)")
        for finding in findings:
            typer.echo(f"  [{finding.code}] {finding.message}")
        raise typer.Exit(1)
    checked = "README.md, cards/README.md, llms.txt"
    if site_dist is not None:
        checked += f", {site_dist}"
    typer.echo(f"release-gate: PASS ({checked}; scope: {scope_path})")


if __name__ == "__main__":
    app()
