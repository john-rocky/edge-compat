"""edge-lint command line interface.

Exit-code contract (stable, CI-native):

- `0` clean (no `--fail-on` rule triggered)
- `1` findings triggered a `--fail-on` rule
      (`--fail-on fallback` triggers on `fallback`, `incorrect`, and `crash`
      verdicts; `--fail-on partitions:N` triggers when partition count > N)
- `2` usage or parse error (unreadable model, invalid matrix snapshot,
      backend mismatch, contradictory matrix entries)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from litert_compat.freshness.releases import (
    ReleasesError,
    load_releases_if_configured,
    staleness_warning,
)
from litert_compat.lint.classify import classify_model
from litert_compat.lint.report import build_report, render_markdown, render_text
from litert_compat.matrix.canonical import canonical_dumps
from litert_compat.matrix.query import Matrix, MatrixLookupTieError
from litert_compat.matrix.validate import MatrixValidationError
from litert_compat.parser.litertlm import (
    LitertlmParseError,
    extract_tflite_model,
    is_litertlm,
)
from litert_compat.parser.reader import TfliteParseError, parse_tflite

app = typer.Typer(
    add_completion=False,
    help="Static delegate-compatibility linter for .tflite models (matrix-backed).",
)

FAIL_ON_STATUSES = frozenset({"fallback", "incorrect", "crash"})


def _parse_fail_on(rules: list[str]) -> tuple[bool, int | None]:
    """-> (fail on fallback/incorrect/crash, max allowed partitions or None)."""
    on_fallback = False
    max_partitions: int | None = None
    for rule in rules:
        if rule == "fallback":
            on_fallback = True
        elif rule.startswith("partitions:"):
            try:
                value = int(rule.removeprefix("partitions:"))
            except ValueError:
                value = -1
            if value < 0:
                typer.echo(f"error: invalid --fail-on rule {rule!r}", err=True)
                raise typer.Exit(2)
            max_partitions = value if max_partitions is None else min(max_partitions, value)
        else:
            typer.echo(
                f"error: invalid --fail-on rule {rule!r} "
                "(expected 'fallback' or 'partitions:N')",
                err=True,
            )
            raise typer.Exit(2)
    return on_fallback, max_partitions


def _triggered(report: dict[str, Any], on_fallback: bool, max_partitions: int | None) -> list[str]:
    triggered = []
    if on_fallback:
        hits = [f for f in report["findings"] if f["status"] in FAIL_ON_STATUSES]
        if hits:
            ops = ", ".join(
                f"s{f['subgraph']}/n{f['node_index']} {f['op']} ({f['status']})" for f in hits
            )
            triggered.append(f"fail-on fallback: {ops}")
    count = report["summary"]["partition_count"]
    if max_partitions is not None and count > max_partitions:
        triggered.append(f"fail-on partitions:{max_partitions}: model has {count} partitions")
    return triggered


@app.command()
def lint(
    model: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Model file (.tflite)."),
    ],
    matrix_file: Annotated[
        Path,
        typer.Option(
            "--matrix", exists=True, dir_okay=False, readable=True, help="Matrix JSON snapshot."
        ),
    ],
    backend: Annotated[
        str | None,
        typer.Option(
            help="Backend ID; must match the matrix file's backend. "
            "Defaults to the matrix file's backend."
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable report (lint_report.schema.json).")
    ] = False,
    md_output: Annotated[bool, typer.Option("--md", help="Markdown report.")] = False,
    fail_on: Annotated[
        list[str] | None,
        typer.Option(
            "--fail-on",
            help="Repeatable. 'fallback' (triggers on fallback/incorrect/crash) "
            "or 'partitions:N' (triggers when partition count > N).",
        ),
    ] = None,
    releases: Annotated[
        Path | None,
        typer.Option(
            "--releases",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Latest-known-release registry (data/releases.json). Emits a soft "
            "staleness warning when the matrix snapshot's litert_version lags the "
            "latest known release; warning only, verdicts unchanged. Defaults to "
            "data/releases.json when that file exists.",
        ),
    ] = None,
    rules_dir: Annotated[
        Path | None,
        typer.Option(
            "--rules",
            exists=True,
            file_okay=False,
            help="edge-fix transform-rules directory. Rewrite hints whose "
            "transform_id resolves to a rule here are annotated 'fixable by "
            "edge-fix' in the text/markdown formats (with the exact command); "
            "the JSON report is unchanged — it carries transform_id verbatim.",
        ),
    ] = None,
) -> None:
    """Statically lint MODEL against a delegate compatibility matrix snapshot."""
    if json_output and md_output:
        typer.echo("error: --json and --md are mutually exclusive", err=True)
        raise typer.Exit(2)
    on_fallback, max_partitions = _parse_fail_on(fail_on or [])

    try:
        matrix = Matrix.load(matrix_file)
    except MatrixValidationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ValueError as exc:
        typer.echo(f"error: {matrix_file}: not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc

    if backend is not None and backend != matrix.backend:
        typer.echo(
            f"error: --backend {backend} does not match matrix backend "
            f"{matrix.backend} ({matrix_file}); one snapshot file per "
            "(backend x litert_version) — pick the matching snapshot",
            err=True,
        )
        raise typer.Exit(2)

    model_bytes = model.read_bytes()
    model_path_echo = str(model)
    if is_litertlm(model_bytes):
        # LiteRT-LM container: lint the embedded TFLite model statically
        # (delegate-level analysis only — E2E bundle audits stay upstream).
        # The echoed path names the section so no report can pass container
        # analysis off as whole-file analysis; sha256 is of the analyzed bytes.
        try:
            model_bytes, section = extract_tflite_model(model_bytes)
        except LitertlmParseError as exc:
            typer.echo(f"error: {model}: {exc}", err=True)
            raise typer.Exit(2) from exc
        suffix = f"::TFLiteModel@section{section.index}"
        if section.model_type:
            suffix += f"({section.model_type})"
        model_path_echo += suffix
    try:
        parsed = parse_tflite(model_bytes)
    except TfliteParseError as exc:
        typer.echo(f"error: {model}: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        classified = classify_model(parsed, matrix)
    except MatrixLookupTieError as exc:
        typer.echo(f"error: matrix data defect: {exc}", err=True)
        raise typer.Exit(2) from exc

    report = build_report(
        classified,
        matrix,
        model_path=model_path_echo,
        model_bytes=model_bytes,
        subgraph_count=len(parsed.subgraphs),
        matrix_path=str(matrix_file),
    )

    resolved_rule_ids: frozenset[str] = frozenset()
    if rules_dir is not None:
        from litert_compat.fix.rules import RuleError, load_rules

        try:
            resolved_rule_ids = frozenset(rule.id for rule in load_rules(rules_dir))
        except RuleError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc

    if json_output:
        if rules_dir is not None:
            typer.echo(
                "note: --rules affects the human formats only; the JSON report "
                "carries transform_id verbatim",
                err=True,
            )
        typer.echo(canonical_dumps(report), nl=False)
    elif md_output:
        typer.echo(
            render_markdown(
                report,
                rules_dir=str(rules_dir) if rules_dir is not None else None,
                resolved_rule_ids=resolved_rule_ids,
            ),
            nl=False,
        )
    else:
        typer.echo(
            render_text(
                report,
                rules_dir=str(rules_dir) if rules_dir is not None else None,
                resolved_rule_ids=resolved_rule_ids,
            ),
            nl=False,
        )

    # Staleness surfacing (Phase 11): soft warning on stderr only — the
    # report, the verdicts, and the exit code are never changed by it.
    try:
        known_releases = load_releases_if_configured(releases)
    except ReleasesError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    warning = staleness_warning(matrix.backend, matrix.litert_version, known_releases)
    if warning is not None:
        typer.echo(f"warning: {warning}", err=True)

    triggered = _triggered(report, on_fallback, max_partitions)
    if triggered:
        for line in triggered:
            typer.echo(f"FAIL {line}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
