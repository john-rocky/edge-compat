"""`edge-card enrich` — Phase 6.

Covers the trap rule's zero-matrix-writes guarantee, 1:1 field mapping from
sweep records, byte-determinism across reruns, demo_url preservation, the
skipped-not-failed path for sweep results without a card, untouched cards
without sweep results, and the nothing-written failure path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.cards.cli import app as cards_app
from litert_compat.cards.index import browser_status
from litert_compat.matrix.canonical import write_canonical

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
WEB_CARDS = (("example-web-a", "web_a"), ("example-web-b", "web_b"))

_RECORD_FIELDS = (
    "backend", "loads", "runs", "full_delegation", "output_match",
    "max_rel_diff", "latency_p50_ms", "env", "date", "provenance",
)


def _workspace(tmp_path: Path) -> Path:
    """Repo-shaped tree: un-enriched web cards, one unrelated card, sweep
    results, and a populated data/matrix/ (for the zero-matrix-writes check)."""
    ws = tmp_path / "ws"
    cards = ws / "cards"
    for model_id, stem in WEB_CARDS:
        result = runner.invoke(
            cards_app,
            [
                "build",
                "--model", str(EXAMPLES / f"model_{stem}_example.tflite"),
                "--meta", str(EXAMPLES / f"meta_{stem}_example.yaml"),
                "-o", str(cards / model_id),
            ],
        )
        assert result.exit_code == 0, result.output
    shutil.copytree(REPO_ROOT / "cards" / "example-tiny-clean", cards / "example-tiny-clean")
    shutil.copytree(EXAMPLES / "sweep", ws / "data" / "examples" / "sweep")
    (ws / "data" / "matrix").mkdir(parents=True)
    shutil.copy(
        EXAMPLES / "matrix_example.json",
        ws / "data" / "matrix" / "gpu_mldrift__0.0.0-example.json",
    )
    return ws


def _enrich(ws: Path) -> Any:
    return runner.invoke(
        cards_app,
        ["enrich", "--sweep", str(ws / "data" / "examples" / "sweep"), str(ws / "cards")],
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_enrich_merges_sweep_records_1_to_1(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    result = _enrich(ws)
    assert result.exit_code == 0, result.output

    sweep = json.loads(
        (ws / "data" / "examples" / "sweep" / "example-web-a.json").read_text(encoding="utf-8")
    )
    card = json.loads((ws / "cards" / "example-web-a" / "card.json").read_text(encoding="utf-8"))
    assert card["schema_version"] == "1.1"
    browser = card["browser"]
    assert browser["demo_url"] is None
    assert browser["sweep_source"] == "data/examples/sweep/example-web-a.json"
    assert len(browser["backends"]) == len(sweep["results"])
    for card_record, sweep_record in zip(browser["backends"], sweep["results"], strict=True):
        assert set(card_record) == set(_RECORD_FIELDS)
        for field in _RECORD_FIELDS:
            assert card_record[field] == sweep_record[field], field

    card_md = (ws / "cards" / "example-web-a" / "CARD.md").read_text(encoding="utf-8")
    assert "## Browser (LiteRT.js)" in card_md
    assert "example-dev-mac-arm64" in card_md  # latency is env-labeled

    index_doc = json.loads((ws / "cards" / "index.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in index_doc["models"]}
    assert by_id["example-web-a"]["browser"]["statuses"] == {
        "wasm_xnnpack": "pass",
        "webgpu_mldrift": "pass",
    }
    assert by_id["example-tiny-clean"]["browser"] is None
    llms = (ws / "llms.txt").read_text(encoding="utf-8")
    assert "data/examples/sweep/example-web-a.json" in llms


def test_enrich_performs_zero_matrix_writes(tmp_path: Path) -> None:
    """The trap rule: enrichment never creates, edits, or removes matrix data —
    every byte under data/ (snapshots AND sweep inputs) is untouched."""
    ws = _workspace(tmp_path)
    data_before = _tree_bytes(ws / "data")
    result = _enrich(ws)
    assert result.exit_code == 0, result.output
    assert _tree_bytes(ws / "data") == data_before


def test_enrich_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    assert _enrich(ws).exit_code == 0
    first = _tree_bytes(ws)
    assert _enrich(ws).exit_code == 0
    assert _tree_bytes(ws) == first


def test_enrich_skips_sweep_results_without_a_card(tmp_path: Path) -> None:
    """example-broken has a sweep result but no card: noted, skipped, exit 0 —
    a sweep may cover models before their cards exist."""
    ws = _workspace(tmp_path)
    result = _enrich(ws)
    assert result.exit_code == 0, result.output
    assert "no card for model 'example-broken'" in result.output
    assert not (ws / "cards" / "example-broken").exists()


def test_enrich_leaves_cards_without_sweep_results_untouched(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    before = _tree_bytes(ws / "cards" / "example-tiny-clean")
    assert _enrich(ws).exit_code == 0
    assert _tree_bytes(ws / "cards" / "example-tiny-clean") == before


def test_enrich_preserves_existing_demo_url(tmp_path: Path) -> None:
    """Phase 7 fills demo_url; re-running enrich must not clobber it."""
    ws = _workspace(tmp_path)
    assert _enrich(ws).exit_code == 0
    card_path = ws / "cards" / "example-web-a" / "card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["browser"]["demo_url"] = "https://example.invalid/demo/example-web-a"
    write_canonical(card, card_path)
    assert _enrich(ws).exit_code == 0
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["browser"]["demo_url"] == "https://example.invalid/demo/example-web-a"


def test_enrich_invalid_sweep_writes_nothing(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "data" / "examples" / "sweep" / "bad.json").write_text("{not json", encoding="utf-8")
    before = _tree_bytes(ws / "cards")
    result = _enrich(ws)
    assert result.exit_code == 1
    assert "nothing written" in result.output
    assert _tree_bytes(ws / "cards") == before
    assert not (ws / "llms.txt").exists()


def test_enrich_rejects_model_id_filename_mismatch(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    sweep_dir = ws / "data" / "examples" / "sweep"
    shutil.copy(sweep_dir / "example-web-a.json", sweep_dir / "example-renamed.json")
    result = _enrich(ws)
    assert result.exit_code == 1
    assert "example-renamed" in result.output


def test_browser_status_vocabulary() -> None:
    def record(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "loads": True, "runs": True, "full_delegation": True, "output_match": True,
        }
        return {**base, **overrides}

    assert browser_status(record(loads=False, runs=False)) == "load_failed"
    assert browser_status(record(runs=False)) == "run_failed"
    assert browser_status(record(output_match=False)) == "output_mismatch"
    assert browser_status(record(full_delegation=False)) == "fallback"
    assert browser_status(record()) == "pass"
    # the wasm_xnnpack reference: comparison and delegation not applicable
    assert browser_status(record(full_delegation=None, output_match=None)) == "pass"
