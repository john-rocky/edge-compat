"""Cards <-> device-run snapshots: the replace trap and its guard (DECISIONS
#161 / #165).

`edge-card enrich` rebuilds a card's device block from exactly the snapshots
it is given, so a snapshot left off the command line silently removed the
rows it contributed — three times, with no test going red, because nothing
read the real `data/device_runs/` against the real `cards/`. These tests do:
the cell-wise newest selection, the drop refusal, `--device-runs-root`,
`device-run validate --cards`, and the committed catalog itself.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from helpers import example_doc, example_record, write_snapshot
from litert_compat.cards.cli import app as cards_app
from litert_compat.cards.enrich import device_cells_dropped, device_coverage_findings
from litert_compat.cards.index import collect_cards
from litert_compat.cli import app as compat_app
from litert_compat.device_runs.records import (
    DeviceRunError,
    discover_device_run_snapshots,
    record_label,
    select_device_runs,
)
from litert_compat.matrix.canonical import canonical_dumps, load_json

runner = CliRunner()

DEVICE_RUNS = REPO_ROOT / "data" / "device_runs"
EXAMPLE_DEVICE_RUNS = REPO_ROOT / "data" / "examples" / "device_runs"
EXAMPLE_SNAPSHOT = EXAMPLE_DEVICE_RUNS / "0.0.0-example" / "2026-01-01"
EXAMPLE_FILE = "example-tiny-clean__example-phone.json"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _card_cells(card_json: Path) -> set[tuple[str, str, str]]:
    """(device, label, snapshot dir) for every device record on a card."""
    return {
        (e["device"], record_label(e["run"]), str(Path(e["source"]).parent.name))
        for e in load_json(card_json)["device"]["records"]
    }


# --- selection ---------------------------------------------------------------


def _mixed_history(root: Path) -> None:
    """One model on one device, measured across four snapshots: the same cell
    re-measured on a newer runtime, a new prompt-length cell, and a
    non-numeric runtime version dated after everything else."""
    write_snapshot(
        root, "0.1.0", "2026-01-01",
        [example_doc("m", "d", [
            example_record("gpu", runtime_version="0.1.0", decode=10.0,
                           metrics={"prefill_tokens": 19}),
            example_record("cpu", runtime_version="0.1.0", decode=5.0),
        ])],
    )
    write_snapshot(
        root, "0.2.0", "2026-02-01",
        [example_doc("m", "d", [
            example_record("gpu", runtime_version="0.2.0", decode=20.0,
                           metrics={"prefill_tokens": 19}),
        ])],
    )
    write_snapshot(
        root, "0.2.0", "2026-03-01",
        [example_doc("m", "d", [
            example_record("gpu", runtime_version="0.2.0", decode=30.0,
                           metrics={"prefill_tokens": 205}),
        ])],
    )
    write_snapshot(
        root, "gallery-1.0", "2026-04-01",
        [example_doc("m", "d", [
            example_record("cpu", runtime_version="gallery-1.0", decode=7.0),
        ])],
    )


def test_select_device_runs_is_cell_wise_newest(tmp_path: Path) -> None:
    _mixed_history(tmp_path / "runs")
    selected = select_device_runs(discover_device_run_snapshots(tmp_path / "runs"))
    cells = {
        (f"{path.parent.parent.name}/{path.parent.name}", record_label(r), r["decode_tokens_per_s"])
        for path, doc in selected
        for r in doc["results"]
    }
    assert cells == {
        # the 0.1.0 gpu@19tok row is superseded by the 0.2.0 re-measurement...
        ("0.2.0/2026-02-01", "gpu@19tok", 20.0),
        # ...the 205-token row is a different cell and coexists (#162)...
        ("0.2.0/2026-03-01", "gpu@205tok", 30.0),
        # ...and cpu stays from 0.1.0: a non-numeric version sorts before every
        # numeric one, whatever its date, so the gallery row loses.
        ("0.1.0/2026-01-01", "cpu", 5.0),
    }
    # A file whose every record was superseded is not returned; a file that
    # kept one of two records is returned cut down to the survivor.
    assert all(doc["results"] for _, doc in selected)
    assert not any("gallery-1.0" in str(path) for path, _ in selected)
    first = next(doc for path, doc in selected if path.parent.name == "2026-01-01")
    assert [r["accelerator"] for r in first["results"]] == ["cpu"]


def test_select_device_runs_refuses_a_cell_measured_under_two_runtimes(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    write_snapshot(
        root, "0.1.0", "2026-01-01",
        [example_doc("m", "d", [example_record("npu", runtime="litert-lm",
                                                 runtime_version="0.1.0")])],
    )
    write_snapshot(
        root, "2.0.0", "2026-01-02",
        [example_doc("m", "d", [example_record("npu", runtime="litert",
                                                 runtime_version="2.0.0")])],
    )
    with pytest.raises(DeviceRunError, match="not comparable"):
        select_device_runs(discover_device_run_snapshots(root))


def test_device_cells_dropped_names_the_lost_cells() -> None:
    card = load_json(REPO_ROOT / "cards" / "example-tiny-clean" / "card.json")
    kept = {**card, "device": {"records": card["device"]["records"][:1]}}
    dropped = device_cells_dropped(card, kept)
    assert dropped == [
        ("example-phone", "npu_qnn",
         "data/examples/device_runs/0.0.0-example/2026-01-01/" + EXAMPLE_FILE),
    ]
    assert device_cells_dropped(card, card) == []
    assert device_cells_dropped({**card, "device": None}, card) == []


# --- enrich: the guard and --device-runs-root ---------------------------------


def _workspace(tmp_path: Path) -> Path:
    """The example cards (the only ones the example snapshots back) plus the
    example snapshot root, laid out repo-shaped so card sources resolve."""
    ws = tmp_path / "ws"
    (ws / "data" / "matrix").mkdir(parents=True)
    for card_dir in sorted((REPO_ROOT / "cards").glob("example-*")):
        shutil.copytree(card_dir, ws / "cards" / card_dir.name)
    shutil.copytree(EXAMPLE_DEVICE_RUNS, ws / "data" / "examples" / "device_runs")
    return ws


def _partial_remeasurement(ws: Path) -> Path:
    """A later example snapshot that re-measures only cpu_xnnpack of
    example-tiny-clean — the shape that, passed alone, drops npu_qnn."""
    doc = load_json(EXAMPLE_SNAPSHOT / EXAMPLE_FILE)
    doc["results"] = [r for r in doc["results"] if r["accelerator"] == "cpu_xnnpack"]
    doc["results"][0]["date"] = "2026-01-03"
    later = ws / "data" / "examples" / "device_runs" / "0.0.0-example" / "2026-01-03"
    later.mkdir()
    (later / EXAMPLE_FILE).write_text(canonical_dumps(doc), encoding="utf-8")
    return later


def test_enrich_refuses_the_replace_trap_unless_the_drop_is_stated(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    later = _partial_remeasurement(ws)
    before = _tree_bytes(ws)

    result = runner.invoke(cards_app, ["enrich", "--device-runs", str(later), str(ws / "cards")])
    assert result.exit_code == 1, result.output
    assert "would be dropped" in result.stderr
    assert "example-tiny-clean: example-phone npu_qnn" in result.stderr
    assert "its snapshot was not passed" in result.stderr
    assert _tree_bytes(ws) == before  # nothing written, not even the catalog outputs

    result = runner.invoke(
        cards_app, ["enrich", "--device-runs", str(later), "--allow-drop", str(ws / "cards")]
    )
    assert result.exit_code == 0, result.output
    assert "dropping device cells" in result.stderr
    assert _card_cells(ws / "cards" / "example-tiny-clean" / "card.json") == {
        ("example-phone", "cpu_xnnpack", "2026-01-03"),
    }


def test_enrich_from_root_needs_no_hand_picked_snapshots(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _partial_remeasurement(ws)
    root = ws / "data" / "examples" / "device_runs"
    args = ["enrich", "--device-runs-root", str(root), str(ws / "cards")]

    result = runner.invoke(cards_app, args)
    assert result.exit_code == 0, result.output
    # cpu_xnnpack from the newer snapshot, npu_qnn kept from the older one.
    assert _card_cells(ws / "cards" / "example-tiny-clean" / "card.json") == {
        ("example-phone", "cpu_xnnpack", "2026-01-03"),
        ("example-phone", "npu_qnn", "2026-01-01"),
    }
    # The other example card, backed by the other snapshot, is untouched.
    assert (ws / "cards" / "example-tiny-mixed" / "card.json").read_bytes() == (
        REPO_ROOT / "cards" / "example-tiny-mixed" / "card.json"
    ).read_bytes()
    first = _tree_bytes(ws)
    assert runner.invoke(cards_app, args).exit_code == 0
    assert _tree_bytes(ws) == first  # byte-stable rerun

    result = runner.invoke(
        cards_app,
        ["enrich", "--device-runs-root", str(root), "--device-runs", str(EXAMPLE_SNAPSHOT),
         str(ws / "cards")],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


# --- device-run validate --cards --------------------------------------------


def test_validate_cards_reports_every_kind_of_drift(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    root = ws / "data" / "examples" / "device_runs"
    args = ["device-run", "validate", str(root), "--cards", str(ws / "cards")]

    result = runner.invoke(compat_app, args)
    assert result.exit_code == 0, result.output
    assert "card(s) match the snapshots" in result.output

    # A newer snapshot measured a cell the card still shows from the older one.
    _partial_remeasurement(ws)
    result = runner.invoke(compat_app, args)
    assert result.exit_code == 1, result.output
    assert "example-tiny-clean: example-phone cpu_xnnpack is from" in result.output
    assert "the newest measurement is" in result.output

    # The library call names the other drifts the CLI would print the same way.
    cards = collect_cards(ws / "cards")
    edited = root / "0.0.0-example" / "2026-01-03" / EXAMPLE_FILE
    doc = load_json(edited)
    doc["results"][0]["decode_tokens_per_s"] = 1.0
    edited.write_text(canonical_dumps(doc), encoding="utf-8")
    shutil.rmtree(root / "0.0.0-example" / "2026-01-01")
    findings = device_coverage_findings(cards, [root], ws)
    assert any("cites data/examples/device_runs/0.0.0-example/2026-01-01/" in f
               and "does not exist" in f for f in findings)
    assert any("npu_qnn cites" in f and "does not exist" in f for f in findings)
    # Rows citing files outside the roots are judged for existence only.
    outside = device_coverage_findings(cards, [tmp_path / "elsewhere"], ws)
    assert all("does not exist" in f for f in outside)

    result = runner.invoke(
        compat_app,
        ["device-run", "validate", str(edited), "--cards", str(ws / "cards")],
    )
    assert result.exit_code == 2  # --cards needs a root


# --- the committed catalog ---------------------------------------------------


def test_committed_cards_are_the_cell_wise_newest_of_every_snapshot() -> None:
    """The invariant the three silent drops violated: every cell any
    snapshot measures for a carded model is on that card, from the newest
    snapshot that measured it, verbatim."""
    cards = collect_cards(REPO_ROOT / "cards")
    findings = device_coverage_findings(cards, [DEVICE_RUNS, EXAMPLE_DEVICE_RUNS], REPO_ROOT)
    assert findings == []


def test_committed_catalog_outputs_are_the_root_enrichment(tmp_path: Path) -> None:
    """Enriching a copy of the repo from the two roots reproduces the
    committed cards, index, README and llms.txt byte for byte — so a
    forgotten `llms.txt` (2ad0039 -> b84738c) or a card rebuilt without
    re-enrichment also goes red here."""
    ws = tmp_path / "ws"
    shutil.copytree(REPO_ROOT / "cards", ws / "cards")
    shutil.copytree(DEVICE_RUNS, ws / "data" / "device_runs")
    shutil.copytree(EXAMPLE_DEVICE_RUNS, ws / "data" / "examples" / "device_runs")
    shutil.copytree(REPO_ROOT / "data" / "matrix", ws / "data" / "matrix")
    result = runner.invoke(
        cards_app,
        [
            "enrich",
            "--device-runs-root", str(ws / "data" / "device_runs"),
            "--device-runs-root", str(ws / "data" / "examples" / "device_runs"),
            str(ws / "cards"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _tree_bytes(ws / "cards") == _tree_bytes(REPO_ROOT / "cards")
    assert (ws / "llms.txt").read_bytes() == (REPO_ROOT / "llms.txt").read_bytes()
