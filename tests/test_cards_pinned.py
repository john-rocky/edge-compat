"""Committed generated outputs are byte-pinned to regeneration.

The committed cards/, llms.txt, and example lint reports ARE tool output —
these tests fail if the tools drift from what is checked in (same discipline
as the byte-pinned matrix example and .tflite fixtures).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.cards.cli import app as cards_app
from litert_compat.lint.cli import app as lint_app

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
COMMITTED_CARDS = ("example-tiny-clean", "example-tiny-mixed")  # built by build-all
WEB_CARDS = (("example-web-a", "web_a"), ("example-web-b", "web_b"))  # build + enrich
ALL_COMMITTED_CARDS = COMMITTED_CARDS + tuple(model_id for model_id, _ in WEB_CARDS)


@pytest.mark.parametrize("fixture", ["clean", "mixed"])
def test_example_lint_reports_pinned_to_lint_output(
    fixture: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """data/examples/lint_*_example.json is exactly `edge-lint --json` output."""
    monkeypatch.chdir(REPO_ROOT)  # reports embed the relative paths passed on the CLI
    result = runner.invoke(
        lint_app,
        [
            f"data/examples/model_{fixture}_example.tflite",
            "--matrix", "data/examples/matrix_example.json",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    committed = (EXAMPLES / f"lint_{fixture}_example.json").read_text(encoding="utf-8")
    assert result.output == committed


def _device_enrich_args(tmp_path: Path, cards_dir: Path) -> list[str]:
    """The committed tiny cards are build output PLUS Phase 13 device-run
    enrichment from the committed example snapshots; copying the snapshots
    under tmp keeps the repo-relative `source` paths identical."""
    shutil.copytree(
        EXAMPLES / "device_runs", tmp_path / "data" / "examples" / "device_runs"
    )
    args = ["enrich"]
    for date in ("2026-01-01", "2026-01-02"):
        args += [
            "--device-runs",
            str(tmp_path / "data" / "examples" / "device_runs" / "0.0.0-example" / date),
        ]
    return [*args, str(cards_dir)]


def test_committed_cards_pinned_to_build_output(tmp_path: Path) -> None:
    result = runner.invoke(
        cards_app,
        [
            "build-all",
            "--manifest", str(EXAMPLES / "cards_manifest_example.csv"),
            "--out", str(tmp_path / "cards"),
        ],
    )
    assert result.exit_code == 1  # the manifest's deliberately broken third row
    shutil.copytree(REPO_ROOT / "data" / "matrix", tmp_path / "data" / "matrix")
    result = runner.invoke(cards_app, _device_enrich_args(tmp_path, tmp_path / "cards"))
    assert result.exit_code == 0, result.output
    for model_id in COMMITTED_CARDS:
        for filename in ("card.json", "CARD.md"):
            committed = (REPO_ROOT / "cards" / model_id / filename).read_bytes()
            rebuilt = (tmp_path / "cards" / model_id / filename).read_bytes()
            assert committed == rebuilt, f"cards/{model_id}/{filename} drifted from build output"


def committed_card_ids() -> list[str]:
    """Every committed cards/<id>/ directory — the index inputs as they exist,
    so the pin keeps holding as real cards land alongside the example ones."""
    return sorted(p.name for p in (REPO_ROOT / "cards").iterdir() if p.is_dir())


def test_committed_index_and_llms_txt_pinned(tmp_path: Path) -> None:
    for model_id in committed_card_ids():
        shutil.copytree(REPO_ROOT / "cards" / model_id, tmp_path / "cards" / model_id)
    # Mirror the repo's committed snapshots so llms.txt links regenerate identically.
    shutil.copytree(REPO_ROOT / "data" / "matrix", tmp_path / "data" / "matrix")
    result = runner.invoke(cards_app, ["index", str(tmp_path / "cards")])
    assert result.exit_code == 0, result.output
    for committed_path, rebuilt_path in (
        (REPO_ROOT / "cards" / "index.json", tmp_path / "cards" / "index.json"),
        (REPO_ROOT / "cards" / "README.md", tmp_path / "cards" / "README.md"),
        (REPO_ROOT / "llms.txt", tmp_path / "llms.txt"),
    ):
        assert committed_path.read_bytes() == rebuilt_path.read_bytes(), (
            f"{committed_path.relative_to(REPO_ROOT)} drifted from index output"
        )


def test_committed_web_cards_pinned_to_build_plus_enrich_output(tmp_path: Path) -> None:
    """cards/example-web-* and the catalog outputs ARE `edge-card build` +
    `edge-card enrich` output on the committed example sweep results."""
    cards_dir = tmp_path / "cards"
    for model_id, stem in WEB_CARDS:
        result = runner.invoke(
            cards_app,
            [
                "build",
                "--model", str(EXAMPLES / f"model_{stem}_example.tflite"),
                "--meta", str(EXAMPLES / f"meta_{stem}_example.yaml"),
                "-o", str(cards_dir / model_id),
            ],
        )
        assert result.exit_code == 0, result.output
    web_ids = {model_id for model_id, _ in WEB_CARDS}
    for model_id in committed_card_ids():
        if model_id not in web_ids:  # the web pair above is freshly rebuilt
            shutil.copytree(REPO_ROOT / "cards" / model_id, cards_dir / model_id)
    shutil.copytree(EXAMPLES / "sweep", tmp_path / "data" / "examples" / "sweep")
    shutil.copytree(REPO_ROOT / "data" / "matrix", tmp_path / "data" / "matrix")

    result = runner.invoke(
        cards_app,
        ["enrich", "--sweep", str(tmp_path / "data" / "examples" / "sweep"), str(cards_dir)],
    )
    assert result.exit_code == 0, result.output

    for model_id in ALL_COMMITTED_CARDS:
        for filename in ("card.json", "CARD.md"):
            committed = (REPO_ROOT / "cards" / model_id / filename).read_bytes()
            rebuilt = (cards_dir / model_id / filename).read_bytes()
            assert committed == rebuilt, f"cards/{model_id}/{filename} drifted from enrich output"
    for committed_path, rebuilt_path in (
        (REPO_ROOT / "cards" / "index.json", cards_dir / "index.json"),
        (REPO_ROOT / "cards" / "README.md", cards_dir / "README.md"),
        (REPO_ROOT / "llms.txt", tmp_path / "llms.txt"),
    ):
        assert committed_path.read_bytes() == rebuilt_path.read_bytes(), (
            f"{committed_path.relative_to(REPO_ROOT)} drifted from enrich output"
        )
