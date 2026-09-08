"""Release-gate (Phase 12.3) tests: benchmark-scope guard (incl. the Gemma-4
failure path), disclosure-line presence, example-provenance banners, site-dist
checks, and the usage-error contract.

Fixture repos are built in tmp from the committed example card; mutations
(non-example cross_runtime provenance, foreign families) exist only there —
nothing real is committed as data.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.branding import DISCLOSURE_LINE, NAMING_NOTICE, PROJECT_NAME
from litert_compat.cards.render import EXAMPLE_BANNER, render_card_markdown
from litert_compat.cli import app
from litert_compat.matrix.canonical import load_json, write_canonical

runner = CliRunner()

CARD_ID = "example-tiny-clean"


def _mini_repo(
    tmp_path: Path,
    *,
    family: str | None = None,
    cross_provenance: str | None = None,
    disclosure: bool = True,
    banner: bool = True,
) -> Path:
    """A minimal gate-checkable repo built from the committed example card."""
    root = tmp_path / "repo"
    card_dir = root / "cards" / CARD_ID
    card_dir.mkdir(parents=True)
    card: dict[str, Any] = load_json(REPO_ROOT / "cards" / CARD_ID / "card.json")
    if family is not None:
        card["model"]["family"] = family
    if cross_provenance is not None:
        for row in card["cross_runtime"]:
            row["provenance"] = cross_provenance
    write_canonical(card, card_dir / "card.json")
    card_md = render_card_markdown(card)
    if not banner:
        card_md = card_md.replace(EXAMPLE_BANNER + "\n\n", "")
    (card_dir / "CARD.md").write_text(card_md, encoding="utf-8")

    surface_text = f"heading\n\n{DISCLOSURE_LINE}\n" if disclosure else "heading\n"
    (root / "README.md").write_text(surface_text, encoding="utf-8")
    (root / "cards" / "README.md").write_text(surface_text, encoding="utf-8")
    (root / "llms.txt").write_text(surface_text, encoding="utf-8")

    (root / "data").mkdir()
    shutil.copyfile(
        REPO_ROOT / "data" / "release_scope.json", root / "data" / "release_scope.json"
    )
    return root


def _gate(root: Path, *extra: str) -> Any:
    return runner.invoke(app, ["release-gate", "--repo-root", str(root), *extra])


def test_gate_passes_on_the_committed_repo() -> None:
    """The DoD anchor: the real repo, real scope file, real surfaces — green."""
    result = runner.invoke(app, ["release-gate", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_branding_rendered_on_committed_surfaces() -> None:
    """Disclosure + naming single-sourced and present on every text surface."""
    for rel in ("README.md", "cards/README.md", "llms.txt"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert DISCLOSURE_LINE in text, rel
    for rel in ("cards/README.md", "llms.txt"):
        assert NAMING_NOTICE in (REPO_ROOT / rel).read_text(encoding="utf-8"), rel
    assert PROJECT_NAME in NAMING_NOTICE


def test_gemma_scope_failure_path(tmp_path: Path) -> None:
    """Non-example cross_runtime rows on a model outside the Gemma-4-only
    scope fail the gate, naming the card and the scope file."""
    root = _mini_repo(tmp_path, cross_provenance="measured")
    result = _gate(root)
    assert result.exit_code == 1, result.output
    assert "[cross_runtime_scope]" in result.output
    assert f"cards/{CARD_ID}/card.json" in result.output
    assert "release_scope.json" in result.output


def test_scope_allows_listed_family(tmp_path: Path) -> None:
    root = _mini_repo(tmp_path, family="gemma-4", cross_provenance="measured")
    result = _gate(root)
    assert result.exit_code == 0, result.output


def test_example_rows_are_exempt_from_scope(tmp_path: Path) -> None:
    """Example-provenance fixtures are governed by the banner rule, not the
    public-comparison scope (DECISIONS: Phase 12)."""
    root = _mini_repo(tmp_path)  # committed rows are provenance example
    result = _gate(root)
    assert result.exit_code == 0, result.output


def test_missing_disclosure_fails(tmp_path: Path) -> None:
    root = _mini_repo(tmp_path, disclosure=False)
    result = _gate(root)
    assert result.exit_code == 1, result.output
    assert result.output.count("[disclosure_missing]") == 3  # all three surfaces


def test_example_banner_path(tmp_path: Path) -> None:
    """A card carrying example-provenance records without the banner fails."""
    root = _mini_repo(tmp_path, banner=False)
    result = _gate(root)
    assert result.exit_code == 1, result.output
    assert "[example_banner_missing]" in result.output
    assert f"cards/{CARD_ID}/CARD.md" in result.output


def test_site_dist_checks(tmp_path: Path) -> None:
    root = _mini_repo(tmp_path)
    dist = tmp_path / "dist"
    (dist / "models" / CARD_ID).mkdir(parents=True)
    (dist / "index.html").write_text(
        f"<p>{DISCLOSURE_LINE}</p><div>example-provenance pipeline fixtures</div>",
        encoding="utf-8",
    )
    (dist / "models" / CARD_ID / "index.html").write_text("<p>no line</p>", encoding="utf-8")
    result = _gate(root, "--site-dist", str(dist))
    assert result.exit_code == 1, result.output
    assert f"dist/models/{CARD_ID}/index.html" in result.output

    # Index without the fixtures banner while example records are shown.
    (dist / "models" / CARD_ID / "index.html").write_text(
        f"<p>{DISCLOSURE_LINE}</p>", encoding="utf-8"
    )
    (dist / "index.html").write_text(f"<p>{DISCLOSURE_LINE}</p>", encoding="utf-8")
    result = _gate(root, "--site-dist", str(dist))
    assert result.exit_code == 1, result.output
    assert "[example_banner_missing]" in result.output
    assert "dist/index.html" in result.output

    # Fully labeled dist is green.
    (dist / "index.html").write_text(
        f"<p>{DISCLOSURE_LINE}</p><div>pipeline fixtures</div>", encoding="utf-8"
    )
    result = _gate(root, "--site-dist", str(dist))
    assert result.exit_code == 0, result.output

    # A dist with no HTML at all is a usage error, not a pass.
    empty = tmp_path / "empty-dist"
    empty.mkdir()
    result = _gate(root, "--site-dist", str(empty))
    assert result.exit_code == 2, result.output


def test_usage_errors(tmp_path: Path) -> None:
    root = _mini_repo(tmp_path)
    result = _gate(root, "--scope", str(tmp_path / "nope.json"))
    assert result.exit_code == 2, result.output

    bad_scope = tmp_path / "bad_scope.json"
    bad_scope.write_text('{"cross_runtime": {"allowed_model_families": "gemma-4"}}')
    result = _gate(root, "--scope", str(bad_scope))
    assert result.exit_code == 2, result.output

    # An invalid card.json refuses to run the gate (never grades a broken repo).
    (root / "cards" / CARD_ID / "card.json").write_text("{}", encoding="utf-8")
    result = _gate(root)
    assert result.exit_code == 2, result.output
