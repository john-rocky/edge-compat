"""`edge-compat freshness` — Phase 11 freshness commands.

Exit-code contract (stable, consumed by the freshness workflows):

- `freshness delta`:   0 no observed changes · 1 changes recorded · 2 usage error
- `freshness overdue`: 0 nothing overdue · 1 overdue items · 2 usage error

Hard rule: these commands never write under `data/matrix/` (tested).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from litert_compat.device_runs.records import (
    DeviceRunError,
    discover_device_run_snapshots,
    snapshots_by_runtime,
)
from litert_compat.freshness.delta import (
    collect_stale_rules,
    compute_delta,
    compute_device_delta,
    render_device_entry,
    render_entry,
    upsert_entry,
)
from litert_compat.freshness.maintenance import RegisterError, overdue_items, parse_register
from litert_compat.freshness.snapshots import SnapshotError, discover_snapshots
from litert_compat.matrix.canonical import canonical_dumps

freshness_app = typer.Typer(
    no_args_is_help=True,
    help="Freshness system (Phase 11): release deltas and manual-verify reminders.",
)


@freshness_app.command()
def delta(
    sweep_dir: Annotated[
        Path,
        typer.Option("--sweep-dir", file_okay=False, help="Sweep snapshot root."),
    ] = Path("data/sweep"),
    device_runs_dir: Annotated[
        Path,
        typer.Option(
            "--device-runs-dir",
            file_okay=False,
            help="Device-run snapshot root (Phase 13); diffed per runtime.",
        ),
    ] = Path("data/device_runs"),
    log: Annotated[
        Path,
        typer.Option("--log", dir_okay=False, help="Delta log to upsert the entry into."),
    ] = Path("DELTA_LOG.md"),
    latency_threshold_pct: Annotated[
        float,
        typer.Option(
            "--latency-threshold-pct",
            min=0.0,
            help="Report latency/throughput shifts with |change| above this percentage.",
        ),
    ] = 25.0,
    rules_dir: Annotated[
        Path | None,
        typer.Option(
            "--rules-dir",
            file_okay=False,
            help="Transform rules directory; rules flagged stale are listed in the entry.",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the delta JSON instead of markdown.")
    ] = False,
) -> None:
    """Diff the two most recent sweep snapshots — and the two most recent
    device-run snapshots per runtime — and upsert the generated entries into
    DELTA_LOG.md. Deterministic; never hand-edited."""
    try:
        sweep_snapshots = discover_snapshots(sweep_dir)
        device_snapshots = discover_device_run_snapshots(device_runs_dir)
    except (SnapshotError, DeviceRunError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    sweep_pair = sweep_snapshots[-2:] if len(sweep_snapshots) >= 2 else None
    device_pairs = [
        group[-2:]
        for group in snapshots_by_runtime(device_snapshots).values()
        if len(group) >= 2
    ]
    if sweep_pair is None and not device_pairs:
        typer.echo(
            f"{sweep_dir}: {len(sweep_snapshots)} sweep snapshot(s), "
            f"{device_runs_dir}: {len(device_snapshots)} device-run snapshot(s) — "
            "need two of a kind to diff; nothing to do"
        )
        raise typer.Exit(0)

    result: dict[str, Any] = {"device_deltas": []}
    rendered: list[str] = []
    changed = False
    if sweep_pair is not None:
        old, new = sweep_pair
        sweep_delta = compute_delta(
            old,
            new,
            latency_threshold_pct=latency_threshold_pct,
            stale_rules=collect_stale_rules(rules_dir),
        )
        result = {**sweep_delta, "device_deltas": []}
        changed = upsert_entry(log, sweep_delta) or changed
        rendered.append(render_entry(sweep_delta))
        typer.echo(
            f"{log}: entry for {old.litertjs_version}@{old.date} → "
            f"{new.litertjs_version}@{new.date} "
            f"{'updated' if changed else 'already current'}",
            err=True,
        )
    has_changes = bool(result.get("has_changes"))
    for old_dev, new_dev in device_pairs:
        device_delta = compute_device_delta(
            old_dev, new_dev, metric_threshold_pct=latency_threshold_pct
        )
        result["device_deltas"].append(device_delta)
        entry_changed = upsert_entry(log, device_delta)
        changed = changed or entry_changed
        rendered.append(render_device_entry(device_delta))
        typer.echo(
            f"{log}: entry for {device_delta['runtime']} "
            f"{old_dev.runtime_version}@{old_dev.date} → "
            f"{new_dev.runtime_version}@{new_dev.date} "
            f"{'updated' if entry_changed else 'already current'}",
            err=True,
        )
        has_changes = has_changes or device_delta["has_changes"]
    result["has_changes"] = has_changes
    typer.echo(canonical_dumps(result) if json_output else "\n".join(rendered), nl=False)
    raise typer.Exit(1 if has_changes else 0)


@freshness_app.command()
def overdue(
    register: Annotated[
        Path,
        typer.Option("--register", dir_okay=False, help="Manual-verify register."),
    ] = Path("MAINTENANCE.md"),
    as_of: Annotated[
        str | None,
        typer.Option(help="YYYY-MM-DD; defaults to today. Pass it for reproducible output."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable output (consumed by reminders.yml).")
    ] = False,
) -> None:
    """List register items whose re-verify interval has lapsed."""
    try:
        as_of_date = (
            datetime.date.fromisoformat(as_of) if as_of is not None else datetime.date.today()
        )
    except ValueError as exc:
        typer.echo(f"error: --as-of must be YYYY-MM-DD, got {as_of!r}", err=True)
        raise typer.Exit(2) from exc
    try:
        items = parse_register(register)
    except RegisterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    late = overdue_items(items, as_of_date)
    if json_output:
        rows = [
            {
                "id": item.id,
                "item": item.item,
                "interval_days": item.interval_days,
                "last_verified": item.last_verified,
                "notes": item.notes,
            }
            for item in late
        ]
        typer.echo(canonical_dumps(rows), nl=False)
    else:
        for item in late:
            typer.echo(
                f"overdue: {item.id} — {item.item} (last verified {item.last_verified}, "
                f"interval {item.interval_days}d)"
            )
        typer.echo(
            f"{len(late)} overdue of {len(items)} register item(s) "
            f"({sum(1 for i in items if i.status == 'dormant')} dormant) as of {as_of_date}"
        )
    raise typer.Exit(1 if late else 0)
