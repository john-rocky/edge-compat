"""edge-fix command line interface.

Exit-code contract (stable, CI-native):

- `0` fixed (plan applied, improvement gate + verification passed — or a
      viable dry-run plan) or nothing to fix
- `1` unfixed issues remain (`no_rules_matched`) or automatic rollback
      (`no_improvement`, `verification_failed`)
- `2` usage or data error (unreadable model, invalid matrix snapshot or rule
      files, unsupported model features, backend mismatch, `--apply` refused
      because verification is unavailable and `--allow-unverified` was not
      passed)

Safety defaults: dry-run is the default; the input file is never modified in
place; a `fix_report.schema.json`-valid report is always emitted (`--json`).
"""

from __future__ import annotations

import datetime
import importlib.metadata
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from litert_compat.fix.engine import FixEngineError, run_fix
from litert_compat.fix.rules import (
    RuleError,
    TransformRule,
    clear_stale,
    load_rule_file,
    load_rules,
    scaffold_rule,
    set_stale,
)
from litert_compat.matrix.canonical import canonical_dumps
from litert_compat.matrix.query import Matrix, MatrixLookupTieError
from litert_compat.matrix.validate import MatrixValidationError
from litert_compat.parser.reader import TfliteParseError
from litert_compat.probe.runners import Tolerance

_EXIT_CODES = {
    "fixed": 0,
    "nothing_to_fix": 0,
    "no_rules_matched": 1,
    "no_improvement": 1,
    "verification_failed": 1,
}


class _DefaultRunGroup(TyperGroup):
    """`edge-fix MODEL.tflite ...` dispatches to the `run` command — the
    spec's CLI shape — while `edge-fix rules ...` keeps normal subcommands."""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        if args and args[0] not in self.list_commands(ctx):
            args = ["run", *args]
        return super().resolve_command(ctx, args)


app = typer.Typer(
    cls=_DefaultRunGroup,
    no_args_is_help=True,
    add_completion=False,
    help="Diagnosis → repair: apply matrix-linked transform rules to a .tflite "
    "model, prove the lint metric improved, and verify the math didn't change.",
)
rules_app = typer.Typer(no_args_is_help=True, help="Transform rule authoring tools.")
app.add_typer(rules_app, name="rules")


def _load_matrix_or_exit(matrix_file: Path, backend: str | None) -> Matrix:
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
    return matrix


def _summary_line(block: dict[str, Any] | None) -> str:
    if block is None:
        return "-"
    s = block["summary"]
    ops = "op" if s["total_ops"] == 1 else "ops"
    parts = "partition" if s["partition_count"] == 1 else "partitions"
    return (
        f"{s['coverage_ops_pct']}% of {s['total_ops']} {ops} delegated · "
        f"{s['partition_count']} {parts}"
    )


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        f"edge-fix: {report['outcome']} — {report['backend']} @ "
        f"litert {report['litert_version']} ({report['mode'].replace('_', '-')})",
        f"model:  {report['model']['path']} (sha256 {report['model']['sha256'][:12]}…)",
        f"matrix: {report['matrix']['file']} "
        f"(generated {report['matrix']['generated_at']})",
        f"before: {_summary_line(report['before'])}",
    ]
    if report["after"] is not None:
        lines.append(f"after:  {_summary_line(report['after'])}")
    if report["plan"]:
        lines.append("plan:")
        for item in report["plan"]:
            where = (
                f"s{item['subgraph']}/n{item['node_index']}"
                if item["node_index"] is not None
                else "graph I/O"
            )
            lines.append(
                f"  {item['rule_id']} [{item['provenance']}] {item['action']} "
                f"{where}: {item['detail']}"
            )
    if report["io_changes"] is not None:
        before, after = report["io_changes"]["before"], report["io_changes"]["after"]
        lines.append(
            f"io:     inputs {','.join(before['inputs'])} → {','.join(after['inputs'])}; "
            f"outputs {','.join(before['outputs'])} → {','.join(after['outputs'])}"
        )
    for key, label in (("conflicts", "conflicts"), ("refused", "refused")):
        for record in report[key]:
            ids = ", ".join(record["rule_ids"]) if key == "conflicts" else record["rule_id"]
            lines.append(f"{label}: {ids} — {record['reason']}")
    for finding in report["unmatched_findings"]:
        lines.append(
            f"unmatched: s{finding['subgraph']}/n{finding['node_index']} "
            f"{finding['op']} ({finding['status']}) — no rule matched; "
            "reported, not guessed at"
        )
    if report["improvement"] is not None:
        verdict = "improved" if report["improvement"]["improved"] else "NOT improved"
        lines.append(f"re-lint: {verdict} — {report['improvement']['reason']}")
    verification = report["verification"]
    if verification is not None:
        detail = verification["reason"] or (
            f"max abs {verification['max_abs_diff']}, max rel {verification['max_rel_diff']}"
        )
        tol = verification["tolerance"]
        lines.append(
            f"verify: {verification['status']} — {detail} "
            f"(tolerance abs={tol['absolute']} rel={tol['relative']})"
        )
    output = report["output"]
    if output["written"]:
        lines.append(f"output: wrote {output['path']} (sha256 {output['sha256'][:12]}…)")
    elif report["outcome"] == "fixed" and report["mode"] == "dry_run":
        lines.append("output: dry-run — pass --apply --out FIXED.tflite to write")
    return "\n".join(lines) + "\n"


@app.command()
def run(
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
    rules_dir: Annotated[
        Path,
        typer.Option(
            "--rules", file_okay=False,
            help="Directory of transform rule files (<id>.json).",
        ),
    ] = Path("data/transforms"),
    backend: Annotated[
        str | None,
        typer.Option(help="Backend ID; must match the matrix file's backend."),
    ] = None,
    apply_fix: Annotated[
        bool,
        typer.Option("--apply", help="Write the fixed model to --out (dry-run is the default)."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Explicitly request the default dry-run mode.")
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", dir_okay=False, help="Output path for the fixed model (--apply)."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", dir_okay=False,
                     help="Write the fix_report.schema.json report here."),
    ] = None,
    allow_unverified: Annotated[
        bool,
        typer.Option(
            "--allow-unverified",
            help="Permit --apply when numerical verification is unavailable "
            "(runners extra missing). Never overrides a FAILED verification.",
        ),
    ] = False,
    tolerance_abs: Annotated[
        float, typer.Option("--tolerance-abs", help="Absolute numeric tolerance vs the original.")
    ] = 1e-5,
    tolerance_rel: Annotated[
        float, typer.Option("--tolerance-rel", help="Relative numeric tolerance vs the original.")
    ] = 1e-3,
) -> None:
    """Fix MODEL: lint → match rules → plan → apply on a copy → re-lint
    (must improve, else rollback) → numerical verification."""
    if apply_fix and dry_run:
        typer.echo("error: --apply and --dry-run are mutually exclusive", err=True)
        raise typer.Exit(2)
    if apply_fix and out is None:
        typer.echo("error: --apply requires --out FIXED.tflite", err=True)
        raise typer.Exit(2)
    if not apply_fix and out is not None:
        typer.echo("error: --out requires --apply (dry-run never writes)", err=True)
        raise typer.Exit(2)
    if out is not None and out.resolve() == model.resolve():
        typer.echo("error: the input file is never modified in place; "
                   "choose a different --out path", err=True)
        raise typer.Exit(2)

    matrix = _load_matrix_or_exit(matrix_file, backend)
    try:
        rules = load_rules(rules_dir)
    except RuleError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        result = run_fix(
            model_path=model,
            matrix=matrix,
            matrix_path=str(matrix_file),
            rules=rules,
            rules_dir=str(rules_dir),
            mode="apply" if apply_fix else "dry_run",
            tolerance=Tolerance(absolute=tolerance_abs, relative=tolerance_rel),
            allow_unverified=allow_unverified,
            out_path=out,
        )
    except TfliteParseError as exc:
        typer.echo(f"error: {model}: {exc}", err=True)
        raise typer.Exit(2) from exc
    except MatrixLookupTieError as exc:
        typer.echo(f"error: matrix data defect: {exc}", err=True)
        raise typer.Exit(2) from exc
    except FixEngineError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    if json_out is not None:
        json_out.write_text(canonical_dumps(result.report), encoding="utf-8")
    typer.echo(_render_text(result.report), nl=False)
    if result.refusal is not None:
        typer.echo(f"error: {result.refusal}", err=True)
        raise typer.Exit(2)
    raise typer.Exit(_EXIT_CODES[result.report["outcome"]])


@rules_app.command("new")
def rules_new(
    rule_id: Annotated[str, typer.Option("--id", help="Rule id (kebab-case).")],
    action: Annotated[
        str,
        typer.Option(help="One of: replace_op, decompose, insert_cast, io_cast."),
    ],
    op: Annotated[
        str | None, typer.Option(help="Builtin op the rule matches (node actions).")
    ] = None,
    fixture_model: Annotated[
        str, typer.Option(help="Fixture model path, relative to the rule file.")
    ] = "TODO-fixture-model.tflite",
    fixture_matrix: Annotated[
        str, typer.Option(help="Fixture matrix path, relative to the rule file.")
    ] = "TODO-fixture-matrix.json",
    out_dir: Annotated[
        Path, typer.Option("--out-dir", file_okay=False, help="Where to write <id>.json.")
    ] = Path("data/transforms"),
    generated_at: Annotated[
        str | None, typer.Option(help="YYYY-MM-DD; defaults to today.")
    ] = None,
) -> None:
    """Scaffold a rule file. Fill the TODO fields, then `edge-fix rules validate` it."""
    if action not in ("replace_op", "decompose", "insert_cast", "io_cast"):
        typer.echo(f"error: unknown action {action!r}", err=True)
        raise typer.Exit(2)
    path = out_dir / f"{rule_id}.json"
    if path.exists():
        typer.echo(f"error: {path} already exists; refusing to overwrite", err=True)
        raise typer.Exit(2)
    doc = scaffold_rule(
        rule_id=rule_id,
        action=action,
        op=op,
        fixture_model=fixture_model,
        fixture_matrix=fixture_matrix,
        generated_at=generated_at or datetime.date.today().isoformat(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(doc), encoding="utf-8")
    typer.echo(f"{path}: scaffolded {action} rule — fill the TODO fields, "
               "then run `edge-fix rules validate`")


def _reverify_one(rule: TransformRule) -> tuple[str, str | None]:
    """-> (status, reason): 'pass' | 'stale' (measured failure) |
    'unavailable' (no measurement — never flagged) | 'error' (data defect)."""
    model_path = rule.fixture_model_path()
    matrix_path = rule.fixture_matrix_path()
    for label, fixture_path in (("model", model_path), ("matrix", matrix_path)):
        if not fixture_path.is_file():
            return "error", f"declared fixture {label} not found: {fixture_path}"
    try:
        matrix = Matrix.load(matrix_path)
    except (MatrixValidationError, ValueError) as exc:
        return "error", f"fixture matrix invalid: {exc}"
    try:
        result = run_fix(
            model_path=model_path,
            matrix=matrix,
            matrix_path=str(matrix_path),
            rules=[rule],
            rules_dir=str(rule.source.parent),
            mode="dry_run",
            tolerance=Tolerance(),
        )
    except (MatrixLookupTieError, TfliteParseError, FixEngineError, ValueError) as exc:
        # The rule could not be exercised end-to-end on its own fixture — a
        # measured inability to apply, not a guess.
        return "stale", f"fixture dry-run error: {exc}"

    outcome = result.report["outcome"]
    verification = result.report["verification"] or {}
    status = verification.get("status")
    if outcome == "fixed" and status == "passed":
        return "pass", None
    if status in ("skipped", "error") and outcome == "fixed":
        # The fix applied but the numerics could not be measured (runners
        # extra missing or interpreter failure): no measurement, no flag.
        return "unavailable", f"verification {status}: {verification.get('reason')}"
    reason = f"fixture dry-run outcome: {outcome}"
    if result.report["improvement"] is not None and not result.report["improvement"]["improved"]:
        reason += f"; re-lint: {result.report['improvement']['reason']}"
    if status == "failed":
        reason += f"; verification failed: {verification.get('reason') or 'outside tolerance'}"
    return "stale", reason


@rules_app.command("reverify")
def rules_reverify(
    rules_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory of rule files (<id>.json)."),
    ] = Path("data/transforms"),
    runtime_version: Annotated[
        str | None,
        typer.Option(
            "--runtime-version",
            help="Runtime version recorded in stale flags; defaults to the installed "
            "ai-edge-litert version.",
        ),
    ] = None,
    date: Annotated[
        str | None,
        typer.Option(help="YYYY-MM-DD recorded in stale flags; defaults to today. "
                     "Pass it for reproducible output."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable per-rule results.")
    ] = False,
) -> None:
    """Re-run every rule's declared fixture against the current runtime
    (spec 11.4). Failing rules are flagged `stale` in their file — never
    auto-deleted; a passing rule's flag is removed. Exit codes: 0 all pass ·
    1 stale rules · 2 data error or verification unavailable."""
    if runtime_version is None:
        try:
            runtime_version = importlib.metadata.version("ai-edge-litert")
        except importlib.metadata.PackageNotFoundError as exc:
            typer.echo(
                "error: re-verification needs the LiteRT CPU interpreter "
                "(install edge-compat[runners]) — without a measurement, "
                "no rule is ever flagged stale",
                err=True,
            )
            raise typer.Exit(2) from exc
    stale_date = date or datetime.date.today().isoformat()
    try:
        rules = load_rules(rules_dir)
    except RuleError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    rows: list[dict[str, Any]] = []
    for rule in rules:
        status, reason = _reverify_one(rule)
        changed = False
        if status == "pass":
            changed = clear_stale(rule.source)
        elif status == "stale":
            changed = set_stale(
                rule.source,
                {
                    "date": stale_date,
                    "runtime": "ai-edge-litert",
                    "runtime_version": runtime_version,
                    "reason": reason,
                },
            )
        rows.append(
            {"id": rule.id, "status": status, "reason": reason, "file_updated": changed}
        )

    if json_output:
        typer.echo(canonical_dumps(rows), nl=False)
    else:
        for row in rows:
            detail = "" if row["reason"] is None else f" — {row['reason']}"
            flag = " (file updated)" if row["file_updated"] else ""
            typer.echo(f"{row['status']:<11} {row['id']}{detail}{flag}")
        typer.echo(
            f"reverified {len(rows)} rule(s) against ai-edge-litert {runtime_version}: "
            f"{sum(1 for r in rows if r['status'] == 'pass')} pass, "
            f"{sum(1 for r in rows if r['status'] == 'stale')} stale, "
            f"{sum(1 for r in rows if r['status'] == 'unavailable')} unavailable, "
            f"{sum(1 for r in rows if r['status'] == 'error')} error"
        )
    if any(row["status"] in ("error", "unavailable") for row in rows):
        raise typer.Exit(2)
    raise typer.Exit(1 if any(row["status"] == "stale" for row in rows) else 0)


@rules_app.command("validate")
def rules_validate(
    rule_files: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Rule files (<id>.json)."),
    ],
) -> None:
    """Validate rule files: schema + semantic checks, then a dry run against
    each rule's own declared fixture — the fix must succeed end-to-end."""
    failures = 0
    for path in rule_files:
        try:
            rule = load_rule_file(path)
        except RuleError as exc:
            failures += 1
            typer.echo(f"{path}: FAIL")
            for error in exc.errors:
                typer.echo(f"  {error}")
            continue

        model_path = rule.fixture_model_path()
        matrix_path = rule.fixture_matrix_path()
        problems: list[str] = []
        for label, fixture_path in (("model", model_path), ("matrix", matrix_path)):
            if not fixture_path.is_file():
                problems.append(f"declared fixture {label} not found: {fixture_path}")
        if problems:
            failures += 1
            typer.echo(f"{path}: FAIL")
            for problem in problems:
                typer.echo(f"  {problem}")
            continue

        try:
            matrix = Matrix.load(matrix_path)
            result = run_fix(
                model_path=model_path,
                matrix=matrix,
                matrix_path=str(matrix_path),
                rules=[rule],
                rules_dir=str(path.parent),
                mode="dry_run",
                tolerance=Tolerance(),
            )
        except (MatrixValidationError, MatrixLookupTieError, TfliteParseError,
                FixEngineError, ValueError) as exc:
            failures += 1
            typer.echo(f"{path}: FAIL")
            typer.echo(f"  fixture dry-run error: {exc}")
            continue

        outcome = result.report["outcome"]
        verification = result.report["verification"] or {}
        if outcome != "fixed":
            failures += 1
            typer.echo(f"{path}: FAIL")
            typer.echo(f"  fixture dry-run outcome: {outcome} (expected fixed)")
            if result.report["improvement"] is not None:
                typer.echo(f"  re-lint: {result.report['improvement']['reason']}")
            if verification.get("status") == "failed":
                typer.echo(f"  verification: {verification['reason']}")
        elif verification.get("status") == "passed":
            typer.echo(f"{path}: PASS (fix applies, metric improves, verification passed)")
        else:
            typer.echo(
                f"{path}: PASS (fix applies, metric improves; verification "
                f"{verification.get('status')}: {verification.get('reason')})"
            )
    raise typer.Exit(1 if failures else 0)


if __name__ == "__main__":
    app()
