"""Phase 11 freshness core: releases registry, snapshot discovery, the delta
generator, the manual-verify register, staleness warnings, and the hard rule
that freshness automation performs zero matrix writes.

All fixtures are example-provenance synthetic sweep records (data integrity
rule 3); no real measurements are fabricated.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.cli import app as compat_app
from litert_compat.freshness.delta import (
    collect_stale_rules,
    compute_delta,
    render_entry,
    upsert_entry,
)
from litert_compat.freshness.maintenance import (
    RegisterError,
    overdue_items,
    parse_register,
)
from litert_compat.freshness.releases import (
    KnownRelease,
    ReleasesError,
    lags_behind,
    load_releases,
    parse_version,
    staleness_warning,
)
from litert_compat.freshness.snapshots import SnapshotError, discover_snapshots
from litert_compat.lint.cli import app as lint_app
from litert_compat.matrix.canonical import canonical_dumps

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"


# --- fixtures ---------------------------------------------------------------


def sweep_record(
    backend: str = "wasm_xnnpack",
    *,
    loads: bool = True,
    runs: bool = True,
    output_match: bool | None = None,
    full_delegation: bool | None = None,
    latency: float | None = 1.0,
    failure_class: str | None = None,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "loads": loads,
        "runs": runs,
        "output_match": output_match,
        "full_delegation": full_delegation,
        "latency_p50_ms": latency,
        "failure_class": failure_class,
        "provenance": "example",
    }


def sweep_doc(model_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model_id": model_id, "results": records, "provenance_note": "example fixture"}


def write_snapshot(
    root: Path, version: str, date: str, docs: list[dict[str, Any]]
) -> Path:
    snap = root / version / date
    snap.mkdir(parents=True)
    for doc in docs:
        (snap / f"{doc['model_id']}.json").write_text(canonical_dumps(doc), encoding="utf-8")
    return snap


def empty_device_runs(tmp_path: Path) -> Path:
    """An isolated, empty device-run root so delta tests never read the repo's
    real data/device_runs (which holds real snapshots since 2026-08-11)."""
    path = tmp_path / "no_device_runs"
    path.mkdir(exist_ok=True)
    return path


def releases_doc(litert: str = "2.1.6", litertjs: str = "2.5.3") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "releases": {
            "litert": {"version": litert, "checked_at": "2026-08-10", "source": "pypi"},
            "litertjs_core": {"version": litertjs, "checked_at": "2026-08-10", "source": "npm"},
        },
    }


# --- releases registry ------------------------------------------------------


def test_parse_version_numeric_prefix() -> None:
    assert parse_version("2.1.6") == (2, 1, 6)
    assert parse_version("0.1.0-rc1") == (0, 1, 0)
    assert parse_version("0.0.0-example") == (0, 0, 0)
    assert parse_version("10") == (10,)
    assert parse_version("nightly") is None


def test_lags_behind_is_strict_and_never_guesses() -> None:
    assert lags_behind("2.1.5", "2.1.6")
    assert lags_behind("2.1", "2.1.6")
    assert not lags_behind("2.1.6", "2.1.6")
    assert not lags_behind("2.1.7", "2.1.6")
    # Unparsable on either side: no warning is better than a wrong one.
    assert not lags_behind("nightly", "2.1.6")
    assert not lags_behind("2.1.6", "nightly")


def test_load_releases_committed_registry() -> None:
    releases = load_releases(REPO_ROOT / "data" / "releases.json")
    assert set(releases) == {"litert", "litertjs_core", "litertlm"}
    for release in releases.values():
        assert release.version
        assert release.checked_at


def test_axis_for_runtime_maps_device_run_lanes() -> None:
    """Phase 13: the .tflite-on-NPU lane stales against litert, the .litertlm
    lane against the new litertlm axis."""
    from litert_compat.freshness.releases import axis_for_runtime

    assert axis_for_runtime("litert") == "litert"
    assert axis_for_runtime("litert-lm") == "litertlm"


def test_load_releases_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "releases.json"
    path.write_text('{"releases": {"litert": {"version": 2}}}', encoding="utf-8")
    with pytest.raises(ReleasesError):
        load_releases(path)


def test_staleness_warning_per_backend_axis() -> None:
    releases = {
        "litert": KnownRelease("2.1.6", "2026-08-10", "pypi"),
        "litertjs_core": KnownRelease("2.5.3", "2026-08-10", "npm"),
    }
    # Native backend compares against litert; web backend against litertjs_core.
    assert "litert 2.1.5" in (staleness_warning("gpu_mldrift", "2.1.5", releases) or "")
    assert "@litertjs/core 2.5.2" in (staleness_warning("webgpu_mldrift", "2.5.2", releases) or "")
    assert staleness_warning("gpu_mldrift", "2.1.6", releases) is None
    assert staleness_warning("webgpu_mldrift", "2.5.3", releases) is None
    assert staleness_warning("gpu_mldrift", "2.1.5", {}) is None
    # Placeholder versions make no claim about a real runtime: never stale.
    assert staleness_warning("gpu_mldrift", "0.0.0-example", releases) is None


def _versioned_matrix(tmp_path: Path, litert_version: str) -> Path:
    """The example snapshot with a strictly numeric litert_version — the only
    kind the staleness warning fires on."""
    doc = json.loads((EXAMPLES / "matrix_example.json").read_text(encoding="utf-8"))
    doc["litert_version"] = litert_version
    path = tmp_path / "matrix.json"
    path.write_text(canonical_dumps(doc), encoding="utf-8")
    return path


def test_lint_soft_warning_warns_without_changing_report_or_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """11.3: warning only — same report bytes, same exit code, stderr only."""
    model = EXAMPLES / "model_clean_example.tflite"
    matrix = _versioned_matrix(tmp_path, "2.1.5")
    stale = tmp_path / "releases.json"
    stale.write_text(canonical_dumps(releases_doc()), encoding="utf-8")

    # CWD without a data/releases.json: no default registry, no warning.
    monkeypatch.chdir(tmp_path)
    plain = runner.invoke(lint_app, [str(model), "--matrix", str(matrix), "--json"])
    warned = runner.invoke(
        lint_app,
        [str(model), "--matrix", str(matrix), "--json", "--releases", str(stale)],
    )
    assert plain.exit_code == warned.exit_code == 0
    assert plain.stdout == warned.stdout  # the report is byte-identical
    assert "warning:" in warned.stderr
    assert "latest known release is 2.1.6" in warned.stderr
    assert "warning:" not in plain.stderr


def test_lint_warns_by_default_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the repo root the committed data/releases.json applies by default
    for snapshots with real versions; example-versioned snapshots never warn
    (which keeps the Phase 2 golden reports byte-stable)."""
    lagging = _versioned_matrix(tmp_path, "2.1.5")
    monkeypatch.chdir(REPO_ROOT)
    model = EXAMPLES / "model_clean_example.tflite"
    result = runner.invoke(lint_app, [str(model), "--matrix", str(lagging), "--json"])
    assert result.exit_code == 0
    assert "warning: matrix snapshot is verified against litert 2.1.5" in result.stderr

    example = runner.invoke(
        lint_app,
        [str(model), "--matrix", str(EXAMPLES / "matrix_example.json"), "--json"],
    )
    assert example.exit_code == 0
    assert "warning:" not in example.stderr


def test_lint_no_warning_when_current(tmp_path: Path) -> None:
    model = EXAMPLES / "model_clean_example.tflite"
    matrix = EXAMPLES / "matrix_example.json"
    current = tmp_path / "releases.json"
    current.write_text(
        canonical_dumps(releases_doc(litert="0.0.0-example")), encoding="utf-8"
    )
    result = runner.invoke(
        lint_app,
        [str(model), "--matrix", str(matrix), "--json", "--releases", str(current)],
    )
    assert result.exit_code == 0
    assert "warning:" not in result.stderr


# --- snapshot discovery -----------------------------------------------------


def test_discover_snapshots_orders_by_version_then_date(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    write_snapshot(root, "0.10.0", "2026-08-01", [sweep_doc("m", [sweep_record()])])
    write_snapshot(root, "0.2.0", "2026-08-05", [sweep_doc("m", [sweep_record()])])
    write_snapshot(root, "0.10.0", "2026-08-03", [sweep_doc("m", [sweep_record()])])
    snapshots = discover_snapshots(root)
    assert [(s.litertjs_version, s.date) for s in snapshots] == [
        ("0.2.0", "2026-08-05"),
        ("0.10.0", "2026-08-01"),
        ("0.10.0", "2026-08-03"),
    ]


def test_discover_snapshots_empty_and_missing_root(tmp_path: Path) -> None:
    assert discover_snapshots(tmp_path / "absent") == []
    (tmp_path / "sweep").mkdir()
    assert discover_snapshots(tmp_path / "sweep") == []


def test_discover_snapshots_rejects_malformed_layout(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    (root / "0.1.0" / "not-a-date").mkdir(parents=True)
    with pytest.raises(SnapshotError, match="YYYY-MM-DD"):
        discover_snapshots(root)


def test_discover_snapshots_rejects_model_id_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    snap = root / "0.1.0" / "2026-08-01"
    snap.mkdir(parents=True)
    (snap / "wrong-name.json").write_text(
        canonical_dumps(sweep_doc("model-a", [sweep_record()])), encoding="utf-8"
    )
    with pytest.raises(SnapshotError, match="model_id"):
        discover_snapshots(root)


# --- delta ------------------------------------------------------------------


def _two_snapshots(root: Path) -> None:
    write_snapshot(
        root,
        "0.1.0",
        "2026-08-01",
        [
            sweep_doc(
                "model-a",
                [
                    sweep_record("wasm_xnnpack", latency=1.0),
                    sweep_record("webgpu_mldrift", output_match=True,
                                 full_delegation=True, latency=0.5),
                ],
            ),
            sweep_doc("model-gone", [sweep_record("wasm_xnnpack")]),
        ],
    )
    write_snapshot(
        root,
        "0.2.0",
        "2026-08-10",
        [
            sweep_doc(
                "model-a",
                [
                    sweep_record("wasm_xnnpack", latency=2.0),
                    sweep_record(
                        "webgpu_mldrift",
                        loads=False,
                        runs=False,
                        latency=None,
                        failure_class="compile_error",
                    ),
                ],
            ),
            sweep_doc("model-new", [sweep_record("wasm_xnnpack")]),
        ],
    )


def test_compute_delta_reports_all_change_kinds(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    _two_snapshots(root)
    old, new = discover_snapshots(root)
    delta = compute_delta(old, new, latency_threshold_pct=25.0, stale_rules=[])
    assert delta["has_changes"]
    assert delta["regressions"] == [
        {
            "model_id": "model-a",
            "backend": "webgpu_mldrift",
            "old_status": "pass",
            "new_status": "load_failed",
            "failure_class": "compile_error",
        }
    ]
    assert delta["improvements"] == []
    assert delta["latency_shifts"] == [
        {
            "model_id": "model-a",
            "backend": "wasm_xnnpack",
            "old_ms": 1.0,
            "new_ms": 2.0,
            "change_pct": 100.0,
        }
    ]
    assert delta["models_added"] == ["model-new"]
    assert delta["models_removed"] == ["model-gone"]


def test_compute_delta_no_changes(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    docs = [sweep_doc("model-a", [sweep_record("wasm_xnnpack", latency=1.0)])]
    write_snapshot(root, "0.1.0", "2026-08-01", docs)
    write_snapshot(root, "0.2.0", "2026-08-10", docs)
    old, new = discover_snapshots(root)
    delta = compute_delta(old, new, latency_threshold_pct=25.0, stale_rules=[])
    assert not delta["has_changes"]
    assert "No observed changes." in render_entry(delta)


def test_latency_threshold_is_strict(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    write_snapshot(
        root, "0.1.0", "2026-08-01",
        [sweep_doc("m", [sweep_record(latency=1.0)])],
    )
    write_snapshot(
        root, "0.2.0", "2026-08-10",
        [sweep_doc("m", [sweep_record(latency=1.25)])],
    )
    old, new = discover_snapshots(root)
    at_threshold = compute_delta(old, new, latency_threshold_pct=25.0, stale_rules=[])
    assert at_threshold["latency_shifts"] == []  # exactly 25% is not beyond it
    below = compute_delta(old, new, latency_threshold_pct=24.0, stale_rules=[])
    assert len(below["latency_shifts"]) == 1


def test_upsert_entry_appends_then_replaces_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    _two_snapshots(root)
    old, new = discover_snapshots(root)
    delta = compute_delta(old, new, latency_threshold_pct=25.0, stale_rules=[])
    log = tmp_path / "DELTA_LOG.md"

    assert upsert_entry(log, delta)
    first = log.read_bytes()
    assert b"# DELTA_LOG" in first
    assert not upsert_entry(log, delta)  # idempotent: same entry, no byte change
    assert log.read_bytes() == first

    # A changed delta for the same version pair replaces its block in place.
    revised = compute_delta(old, new, latency_threshold_pct=1000.0, stale_rules=[])
    assert upsert_entry(log, revised)
    content = log.read_text(encoding="utf-8")
    assert content.count("## @litertjs/core 0.1.0 → 0.2.0") == 1
    assert "Latency shifts" not in content


def test_delta_cli_exit_codes_and_log(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    _two_snapshots(root)
    log = tmp_path / "DELTA_LOG.md"
    result = runner.invoke(
        compat_app,
        ["freshness", "delta", "--sweep-dir", str(root), "--log", str(log), "--json",
         "--device-runs-dir", str(empty_device_runs(tmp_path))],
    )
    assert result.exit_code == 1, result.output  # changes found
    payload = json.loads(result.stdout)
    assert payload["has_changes"]
    assert log.is_file()

    # Fewer than two snapshots: nothing to diff, exit 0, no log written.
    single = tmp_path / "single"
    write_snapshot(single, "0.1.0", "2026-08-01", [sweep_doc("m", [sweep_record()])])
    result = runner.invoke(
        compat_app,
        ["freshness", "delta", "--sweep-dir", str(single), "--log", str(tmp_path / "x.md"),
         "--device-runs-dir", str(empty_device_runs(tmp_path))],
    )
    assert result.exit_code == 0
    assert not (tmp_path / "x.md").exists()


def test_delta_cli_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    _two_snapshots(root)
    outputs = []
    for run in range(2):
        log = tmp_path / f"log-{run}.md"
        result = runner.invoke(
            compat_app,
            ["freshness", "delta", "--sweep-dir", str(root), "--log", str(log), "--json",
             "--device-runs-dir", str(empty_device_runs(tmp_path))],
        )
        outputs.append((result.stdout, log.read_bytes()))
    assert outputs[0] == outputs[1]


def test_delta_lists_stale_rules(tmp_path: Path) -> None:
    rules = tmp_path / "transforms"
    rules.mkdir()
    src = EXAMPLES / "transforms" / "example-replace-div-floor-div.json"
    doc = json.loads(src.read_text(encoding="utf-8"))
    doc["stale"] = {
        "date": "2026-08-10",
        "runtime": "ai-edge-litert",
        "runtime_version": "9.9.9",
        "reason": "fixture dry-run outcome: no_improvement",
    }
    (rules / src.name).write_text(canonical_dumps(doc), encoding="utf-8")
    shutil.copy(EXAMPLES / "transforms" / "example-decompose-square-mul.json", rules)

    stale = collect_stale_rules(rules)
    assert [r["id"] for r in stale] == ["example-replace-div-floor-div"]

    root = tmp_path / "sweep"
    docs = [sweep_doc("m", [sweep_record(latency=1.0)])]
    write_snapshot(root, "0.1.0", "2026-08-01", docs)
    write_snapshot(root, "0.2.0", "2026-08-10", docs)
    old, new = discover_snapshots(root)
    delta = compute_delta(old, new, latency_threshold_pct=25.0, stale_rules=stale)
    assert delta["has_changes"]  # stale rules alone are a reportable change
    entry = render_entry(delta)
    assert "Transform rules flagged stale (1)" in entry
    assert "example-replace-div-floor-div" in entry


# --- maintenance register ---------------------------------------------------


def test_parse_committed_register() -> None:
    items = parse_register(REPO_ROOT / "MAINTENANCE.md")
    assert items, "committed MAINTENANCE.md must carry at least one register row"
    assert len({i.id for i in items}) == len(items)


def register_text(rows: list[str]) -> str:
    return (
        "# MAINTENANCE\n\nprose\n\n"
        "| id | item | interval_days | status | last_verified | notes |\n"
        "|---|---|---|---|---|---|\n" + "".join(f"{row}\n" for row in rows)
    )


def test_overdue_computation(tmp_path: Path) -> None:
    register = tmp_path / "MAINTENANCE.md"
    register.write_text(
        register_text(
            [
                "| fresh | Recently verified | 30 | active | 2026-08-01 | |",
                "| lapsed | Interval elapsed | 30 | active | 2026-06-01 | |",
                "| unstarted | Never verified | 30 | active | never | |",
                "| sleeping | No real data yet | 30 | dormant | never | |",
            ]
        ),
        encoding="utf-8",
    )
    items = parse_register(register)
    late = overdue_items(items, datetime.date(2026, 8, 10))
    assert [i.id for i in late] == ["lapsed", "unstarted"]


def test_overdue_cli(tmp_path: Path) -> None:
    register = tmp_path / "MAINTENANCE.md"
    register.write_text(
        register_text(["| lapsed | Item | 30 | active | 2026-01-01 | note |"]),
        encoding="utf-8",
    )
    result = runner.invoke(
        compat_app,
        ["freshness", "overdue", "--register", str(register),
         "--as-of", "2026-08-10", "--json"],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == [
        {
            "id": "lapsed",
            "item": "Item",
            "interval_days": 30,
            "last_verified": "2026-01-01",
            "notes": "note",
        }
    ]

    register.write_text(
        register_text(["| ok | Item | 3650 | active | 2026-08-01 | |"]), encoding="utf-8"
    )
    result = runner.invoke(
        compat_app,
        ["freshness", "overdue", "--register", str(register), "--as-of", "2026-08-10"],
    )
    assert result.exit_code == 0


def test_register_parse_errors(tmp_path: Path) -> None:
    register = tmp_path / "MAINTENANCE.md"
    register.write_text("# no table here\n", encoding="utf-8")
    with pytest.raises(RegisterError):
        parse_register(register)
    register.write_text(
        register_text(["| bad | Item | soon | active | never | |"]), encoding="utf-8"
    )
    with pytest.raises(RegisterError, match="interval_days"):
        parse_register(register)


# --- the hard rule: zero matrix writes --------------------------------------


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_freshness_automation_performs_zero_matrix_writes(tmp_path: Path) -> None:
    """Spec Phase 11 Out (hard): no automated writes to op-level matrix
    entries from freshness workflows. Every command the workflows run leaves
    every byte under data/matrix/ untouched."""
    ws = tmp_path / "ws"
    matrix_dir = ws / "data" / "matrix"
    matrix_dir.mkdir(parents=True)
    shutil.copy(
        EXAMPLES / "matrix_example.json", matrix_dir / "gpu_mldrift__0.0.0-example.json"
    )
    shutil.copy(
        EXAMPLES / "matrix_webgpu_mldrift_example.json",
        matrix_dir / "webgpu_mldrift__0.0.0-example.json",
    )
    _two_snapshots(ws / "data" / "sweep")
    register = ws / "MAINTENANCE.md"
    register.write_text(
        register_text(["| item | Thing | 30 | active | never | |"]), encoding="utf-8"
    )
    rules_dir = ws / "data" / "transforms"
    shutil.copytree(EXAMPLES / "transforms", rules_dir)

    device_runs = ws / "data" / "device_runs"
    shutil.copytree(EXAMPLES / "device_runs", device_runs)

    before = _tree_bytes(matrix_dir)
    assert runner.invoke(
        compat_app,
        ["freshness", "delta", "--sweep-dir", str(ws / "data" / "sweep"),
         "--device-runs-dir", str(device_runs),
         "--log", str(ws / "DELTA_LOG.md"), "--rules-dir", str(rules_dir)],
    ).exit_code == 1
    assert runner.invoke(
        compat_app,
        ["freshness", "overdue", "--register", str(register), "--as-of", "2026-08-10"],
    ).exit_code == 1
    # Phase 13 device-run commands honor the same hard rule.
    assert runner.invoke(
        compat_app,
        [
            "device-run", "ingest-gpu-audit",
            str(EXAMPLES / "llm_samples" / "gpu_audit" / "gemma3-1b-official.gpu.log"),
            "--model-id", "gemma3-1b", "--device", "test-mac",
            "--runtime-version", "0.15.0", "--date", "2026-07-23",
            "--out", str(device_runs),
        ],
    ).exit_code == 0
    assert runner.invoke(
        compat_app, ["device-run", "validate", str(device_runs)]
    ).exit_code == 0
    assert _tree_bytes(matrix_dir) == before
