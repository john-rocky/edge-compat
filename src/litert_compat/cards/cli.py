"""edge-card command line interface.

Exit-code contract (stable, CI-native):

- `build`:     0 card written · 1 invalid input data (nothing written) · 2 usage error
- `build-all`: 0 all rows built · 1 >=1 row failed (failures collected and
               reported at the end — never fail-fast) · 2 usage error
- `index`:     0 outputs written · 1 invalid card.json found (nothing written)
               · 2 usage error
- `enrich`:    0 outputs written (sweep results without a card are notes, not
               failures) · 1 invalid sweep result or card.json, or a device
               cell a card carries would be dropped without --allow-drop
               (nothing written) · 2 usage error
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from litert_compat.cards.build import CardBuildError, build_card
from litert_compat.cards.enrich import (
    CardEnrichError,
    SweepLoadError,
    device_cells_dropped,
    enrich_card,
    enrich_card_device,
    load_device_runs,
    load_sweep_results,
    repo_relative,
)
from litert_compat.cards.index import (
    CardIndexError,
    build_index_doc,
    collect_cards,
    matrix_snapshot_files,
    render_index_markdown,
    render_llms_txt,
)
from litert_compat.cards.meta import MetaError, load_meta
from litert_compat.cards.render import render_card_markdown
from litert_compat.cards.schema_io import schema_errors
from litert_compat.device_runs.records import (
    DeviceRunError,
    discover_device_run_snapshots,
    select_device_runs,
)
from litert_compat.matrix.canonical import load_json, write_canonical

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Perf cards: card.json + CARD.md, catalog index, llms.txt.",
)

_MANIFEST_COLUMNS = ("model", "meta", "bench", "lint", "cross")


class _InputError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _load_validated(path: Path, schema_name: str, label: str) -> dict[str, Any]:
    try:
        doc = load_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        raise _InputError([f"{label} {path}: not readable JSON: {exc}"]) from exc
    errors = schema_errors(doc, schema_name)
    if errors:
        raise _InputError([f"{label} {path}: {e}" for e in errors])
    return doc


def _build_one(
    model: Path | None,
    meta: dict[str, Any],
    bench: Path | None,
    lint: Path | None,
    cross_bench: Path | None,
    out_dir: Path,
    remote_artifact: dict[str, Any] | None = None,
) -> None:
    """Build one card into out_dir. Raises _InputError with all defects; writes nothing then."""
    bench_doc = (
        _load_validated(bench, "benchmark_result.schema.json", "bench") if bench else None
    )
    cross_doc = (
        _load_validated(cross_bench, "benchmark_result.schema.json", "cross-bench")
        if cross_bench
        else None
    )
    lint_doc = _load_validated(lint, "lint_report.schema.json", "lint report") if lint else None
    try:
        card = build_card(model, meta, bench_doc, lint_doc, cross_doc, remote_artifact)
    except CardBuildError as exc:
        raise _InputError(exc.errors) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    write_canonical(card, out_dir / "card.json")
    (out_dir / "CARD.md").write_text(render_card_markdown(card), encoding="utf-8")


@app.command()
def build(
    meta: Annotated[
        Path,
        typer.Option(
            "--meta", exists=True, dir_okay=False, readable=True, help="Human-authored meta.yaml."
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output directory for card.json + CARD.md.")
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            "--model", exists=True, dir_okay=False, readable=True,
            help="Model file (.tflite or .litertlm); read as bytes for sha256/size only. "
            "Omit when the published bytes are not local — pass the three "
            "--artifact-* values verbatim from the hosting API instead.",
        ),
    ] = None,
    bench: Annotated[
        Path | None,
        typer.Option(
            "--bench",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Benchmark results JSON (benchmark_result.schema.json).",
        ),
    ] = None,
    lint: Annotated[
        Path | None,
        typer.Option(
            "--lint",
            exists=True,
            dir_okay=False,
            readable=True,
            help="edge-lint --json report (lint_report.schema.json).",
        ),
    ] = None,
    cross_bench: Annotated[
        Path | None,
        typer.Option(
            "--cross-bench",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Cross-runtime results JSON (same format; `runtime` required per record).",
        ),
    ] = None,
    artifact_file: Annotated[
        str | None,
        typer.Option(
            "--artifact-file",
            help="Published artifact file name (remote mode; with --artifact-sha256/--artifact-size-bytes).",
        ),
    ] = None,
    artifact_sha256: Annotated[
        str | None,
        typer.Option(
            "--artifact-sha256",
            help="Artifact sha256, verbatim from the hosting API (e.g. HF LFS). Never hand-computed.",
        ),
    ] = None,
    artifact_size_bytes: Annotated[
        int | None,
        typer.Option(
            "--artifact-size-bytes",
            help="Artifact byte size, verbatim from the hosting API.",
        ),
    ] = None,
) -> None:
    """Build one card: card.json (source of truth) + CARD.md (rendered view)."""
    remote_values = (artifact_file, artifact_sha256, artifact_size_bytes)
    remote_given = any(v is not None for v in remote_values)
    if remote_given and not all(v is not None for v in remote_values):
        typer.echo(
            "build FAILED (nothing written):\n"
            "  remote mode needs all three of --artifact-file / --artifact-sha256 / "
            "--artifact-size-bytes",
            err=True,
        )
        raise typer.Exit(1)
    if (model is None) == (not remote_given):
        typer.echo(
            "build FAILED (nothing written):\n"
            "  pass exactly one of --model or the --artifact-* triple",
            err=True,
        )
        raise typer.Exit(1)
    remote_artifact = (
        {"file": artifact_file, "sha256": artifact_sha256, "size_bytes": artifact_size_bytes}
        if remote_given
        else None
    )
    try:
        meta_doc, warnings = load_meta(meta)
        for warning in warnings:
            typer.echo(f"warning: {warning}", err=True)
        _build_one(model, meta_doc, bench, lint, cross_bench, out, remote_artifact)
    except (_InputError, MetaError) as exc:
        typer.echo("build FAILED (nothing written):", err=True)
        for error in exc.errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{out}: wrote card.json + CARD.md")


def _manifest_paths(row: dict[str, str | None], base: Path) -> dict[str, Path | None]:
    """Resolve a manifest row's cells against the manifest's directory."""
    out: dict[str, Path | None] = {}
    for column in _MANIFEST_COLUMNS:
        value = (row.get(column) or "").strip()
        out[column] = base / value if value else None
    return out


@app.command("build-all")
def build_all(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            help="CSV manifest; paths resolve relative to the manifest file.",
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output root; each card goes to <out>/<model_id>/.")
    ] = Path("cards"),
) -> None:
    """Batch build from a CSV manifest; failures are collected, not fail-fast."""
    with manifest.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [name.strip().lower() for name in reader.fieldnames or []]
        reader.fieldnames = fieldnames
        if "model" not in fieldnames or "meta" not in fieldnames:
            typer.echo(
                f"error: {manifest}: manifest needs a header with 'model' and 'meta' columns",
                err=True,
            )
            raise typer.Exit(2)
        for unknown in sorted(set(fieldnames) - set(_MANIFEST_COLUMNS)):
            typer.echo(f"warning: unknown manifest column {unknown!r} ignored", err=True)
        rows = list(reader)

    base = manifest.parent
    failures: list[tuple[int, str]] = []
    built = 0
    for line_no, row in enumerate(rows, start=2):  # header is line 1
        paths = _manifest_paths(row, base)
        problems = [
            f"{column} file not found: {path}"
            for column, path in paths.items()
            if path is not None and not path.is_file()
        ]
        for column in ("model", "meta"):
            if paths[column] is None:
                problems.append(f"{column} path is required")
        if problems:
            failures.extend((line_no, p) for p in problems)
            continue

        model, meta_path = paths["model"], paths["meta"]
        assert model is not None and meta_path is not None
        try:
            meta_doc, warnings = load_meta(meta_path)
            for warning in warnings:
                typer.echo(f"warning: line {line_no}: {warning}", err=True)
            _build_one(
                model, meta_doc, paths["bench"], paths["lint"], paths["cross"],
                out / meta_doc["model"]["id"],
            )
            built += 1
        except (_InputError, MetaError) as exc:
            failures.extend((line_no, e) for e in exc.errors)

    typer.echo(f"built {built} of {len(rows)} card(s) into {out}/")
    if failures:
        typer.echo(f"{len(failures)} failure(s):", err=True)
        for line_no, message in failures:
            typer.echo(f"  line {line_no}: {message}", err=True)
        raise typer.Exit(1)


def _write_catalog_outputs(cards: list[tuple[str, dict[str, Any]]], cards_dir: Path) -> None:
    """Write cards/index.json, cards/README.md, llms.txt — one code path, so
    `index` and `enrich` can never disagree about the catalog outputs."""
    repo_root = cards_dir.resolve().parent
    index_doc = build_index_doc(cards)
    write_canonical(index_doc, cards_dir / "index.json")
    (cards_dir / "README.md").write_text(render_index_markdown(index_doc), encoding="utf-8")
    llms = render_llms_txt(index_doc, matrix_snapshot_files(repo_root))
    (repo_root / "llms.txt").write_text(llms, encoding="utf-8")


@app.command()
def index(
    cards_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Cards directory (default: cards/)."),
    ] = Path("cards"),
) -> None:
    """Generate cards/README.md, cards/index.json, and llms.txt (in cards_dir's parent)."""
    try:
        cards = collect_cards(cards_dir)
    except CardIndexError as exc:
        typer.echo("index FAILED (nothing written):", err=True)
        for error in exc.errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(1) from exc

    repo_root = cards_dir.resolve().parent
    _write_catalog_outputs(cards, cards_dir)
    typer.echo(
        f"indexed {len(cards)} card(s): {cards_dir / 'index.json'}, "
        f"{cards_dir / 'README.md'}, {repo_root / 'llms.txt'}"
    )


@app.command()
def enrich(
    sweep: Annotated[
        Path | None,
        typer.Option(
            "--sweep",
            exists=True,
            file_okay=False,
            help="Directory of sweep result JSON files (sweep_result.schema.json).",
        ),
    ] = None,
    device_runs: Annotated[
        list[Path] | None,
        typer.Option(
            "--device-runs",
            exists=True,
            file_okay=False,
            help="Device-run snapshot directory (device_run_result.schema.json files); "
            "repeatable — pass one per (runtime x version x date) snapshot.",
        ),
    ] = None,
    device_runs_root: Annotated[
        list[Path] | None,
        typer.Option(
            "--device-runs-root",
            exists=True,
            file_okay=False,
            help="Device-run snapshot ROOT (<root>/<version>/<date>/ below it); repeatable. "
            "Every snapshot under it is read and each (device x accelerator x prompt-length) "
            "cell takes its newest measurement, so no snapshot is hand-picked (DECISIONS "
            "#165). Mutually exclusive with --device-runs.",
        ),
    ] = None,
    allow_drop: Annotated[
        bool,
        typer.Option(
            "--allow-drop",
            help="Write even where a card would lose a device cell it carries today. Without "
            "it such a drop is refused with nothing written — the replace trap (DECISIONS "
            "#161).",
        ),
    ] = False,
    cards_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Cards directory (default: cards/)."),
    ] = Path("cards"),
) -> None:
    """Merge sweep and/or device-run results into cards; regenerate CARD.md, index, llms.txt.

    Model-level enrichment only — NEVER writes matrix entries (the trap rule,
    both editions). Results without a matching card are reported and skipped;
    cards without results are left byte-for-byte untouched. A card's device
    block is rebuilt from exactly the snapshots given, so a cell it carries
    today that those snapshots do not measure would be dropped: that is
    refused unless --allow-drop states it (--device-runs-root reads every
    snapshot and never needs it while the files exist).
    """
    if sweep is None and not device_runs and not device_runs_root:
        typer.echo("error: pass --sweep, --device-runs and/or --device-runs-root", err=True)
        raise typer.Exit(2)
    if device_runs and device_runs_root:
        typer.echo(
            "error: --device-runs and --device-runs-root are mutually exclusive — the root "
            "already selects the newest snapshot per cell",
            err=True,
        )
        raise typer.Exit(2)
    try:
        sweep_results = load_sweep_results(sweep) if sweep is not None else []
        if device_runs_root:
            snapshots = [
                snapshot
                for root in device_runs_root
                for snapshot in discover_device_run_snapshots(root)
            ]
            device_run_files = select_device_runs(snapshots)
        else:
            device_run_files = load_device_runs(device_runs) if device_runs else []
        cards = collect_cards(cards_dir)
    except DeviceRunError as exc:
        typer.echo("enrich FAILED (nothing written):", err=True)
        for line in str(exc).splitlines():
            typer.echo(f"  {line}", err=True)
        raise typer.Exit(1) from exc
    except (SweepLoadError, CardIndexError) as exc:
        typer.echo("enrich FAILED (nothing written):", err=True)
        for error in exc.errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(1) from exc

    repo_root = cards_dir.resolve().parent
    original = dict(cards)
    card_by_id = dict(cards)
    enriched_ids: set[str] = set()
    try:
        for path, sweep_doc in sweep_results:
            model_id = sweep_doc["model_id"]
            if model_id not in card_by_id:
                typer.echo(
                    f"note: {path.name}: no card for model {model_id!r} — skipped", err=True
                )
                continue
            source = repo_relative(path, repo_root)
            card_by_id[model_id] = enrich_card(card_by_id[model_id], sweep_doc, source)
            enriched_ids.add(model_id)

        runs_by_model: dict[str, list[tuple[str, Any]]] = {}
        for path, run_doc in device_run_files:
            model_id = run_doc["model_id"]
            if model_id not in card_by_id:
                typer.echo(
                    f"note: {path.name}: no card for model {model_id!r} — skipped", err=True
                )
                continue
            runs_by_model.setdefault(model_id, []).append(
                (repo_relative(path, repo_root), run_doc)
            )
        for model_id, runs in runs_by_model.items():
            card_by_id[model_id] = enrich_card_device(card_by_id[model_id], runs)
            enriched_ids.add(model_id)
    except CardEnrichError as exc:
        typer.echo("enrich FAILED (nothing written):", err=True)
        for error in exc.errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(1) from exc

    dropped = [
        (model_id, device, label, source)
        for model_id in sorted(enriched_ids)
        for device, label, source in device_cells_dropped(
            original[model_id], card_by_id[model_id]
        )
    ]
    if dropped:
        typer.echo(
            "note: dropping device cells (--allow-drop):"
            if allow_drop
            else "enrich FAILED (nothing written): these device cells would be dropped —",
            err=True,
        )
        for model_id, device, label, source in dropped:
            why = (
                "its snapshot was not passed"
                if (repo_root / source).is_file()
                else "its source file no longer exists"
            )
            typer.echo(f"  {model_id}: {device} {label} from {source} — {why}", err=True)
        if not allow_drop:
            typer.echo(
                "  pass every snapshot that backs a card (or --device-runs-root), or state "
                "the drop with --allow-drop",
                err=True,
            )
            raise typer.Exit(1)

    for model_id in sorted(enriched_ids):
        card = card_by_id[model_id]
        out_dir = cards_dir / model_id
        write_canonical(card, out_dir / "card.json")
        (out_dir / "CARD.md").write_text(render_card_markdown(card), encoding="utf-8")
    _write_catalog_outputs([(d, card_by_id[d]) for d, _ in cards], cards_dir)
    typer.echo(
        f"enriched {len(enriched_ids)} of {len(cards)} card(s) from "
        f"{len(sweep_results)} sweep result(s) and "
        f"{len(device_run_files)} device-run file(s); catalog outputs regenerated"
    )

if __name__ == "__main__":
    app()
