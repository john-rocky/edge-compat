"""Release-gate lint (Phase 12.3): deterministic pre-publication checks.

Run by the clearance-flip runbook (RELEASE.md) and CI before anything goes
public. Three guards:

1. **Benchmark scope** — no card presents a cross-runtime comparison for a
   model outside the approved public scope (`data/release_scope.json`,
   owner-editable data; initially Gemma 4 only per the launch plan).
   Example-provenance rows are pipeline fixtures governed by the
   example-banner rule, not public comparative claims, so the scope guard
   applies to non-example rows.
2. **Disclosure** — the disclosure line (litert_compat.branding) renders on
   every public text surface: README.md, cards/README.md, llms.txt, and,
   with a built site directory, every generated HTML page.
3. **Example banners** — a card carrying example-provenance records must
   render the example banner in its CARD.md, and the site index must show
   its fixtures banner whenever example records are displayed.

Findings are emitted in a fixed check order with sorted file iteration, so
output is deterministic. Invalid inputs (unreadable scope file, invalid
card.json) are usage errors, not findings — the gate refuses rather than
grading a broken repo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_compat.branding import DISCLOSURE_LINE
from litert_compat.cards.index import CardIndexError, collect_cards
from litert_compat.cards.render import EXAMPLE_BANNER, provenance_values
from litert_compat.matrix.canonical import load_json

# Substring of every site example banner variant (index page, demo pages —
# site/src/generator/render.ts); asserted against built output only.
SITE_EXAMPLE_MARKER = "pipeline fixture"

# Public text surfaces, repo-relative (the site pages are covered separately
# via --site-dist because dist/ is generated, not committed).
TEXT_SURFACES = ("README.md", "cards/README.md", "llms.txt")


class ReleaseGateUsageError(ValueError):
    """The gate cannot run: unreadable/invalid scope file or invalid cards."""


@dataclass(frozen=True)
class GateFinding:
    code: str
    message: str


def load_scope(scope_path: Path) -> dict[str, Any]:
    """Load and structurally validate the owner-editable scope file."""
    try:
        doc = load_json(scope_path)
    except OSError as exc:
        raise ReleaseGateUsageError(f"{scope_path}: not readable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseGateUsageError(f"{scope_path}: not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ReleaseGateUsageError(f"{scope_path}: expected a JSON object")
    cross = doc.get("cross_runtime")
    if not isinstance(cross, dict):
        raise ReleaseGateUsageError(f"{scope_path}: missing 'cross_runtime' object")
    for key in ("allowed_model_families", "allowed_model_ids"):
        values = cross.get(key)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ReleaseGateUsageError(
                f"{scope_path}: 'cross_runtime.{key}' must be a list of strings"
            )
    return doc


def _scope_findings(
    cards: list[tuple[str, dict[str, Any]]], scope: dict[str, Any], scope_name: str
) -> list[GateFinding]:
    allowed_families = set(scope["cross_runtime"]["allowed_model_families"])
    allowed_ids = set(scope["cross_runtime"]["allowed_model_ids"])
    findings: list[GateFinding] = []
    for dir_name, card in cards:
        rows = [r for r in card["cross_runtime"] if r["provenance"] != "example"]
        if not rows:
            continue
        model = card["model"]
        if model["family"] in allowed_families or model["id"] in allowed_ids:
            continue
        runtimes = ", ".join(sorted({r["runtime"] for r in rows}))
        findings.append(
            GateFinding(
                "cross_runtime_scope",
                f"cards/{dir_name}/card.json: {len(rows)} non-example cross_runtime "
                f"row(s) ({runtimes}), but model id {model['id']!r} / family "
                f"{model['family']!r} is outside the approved public scope "
                f"({scope_name})",
            )
        )
    return findings


def _disclosure_findings(repo_root: Path) -> list[GateFinding]:
    findings: list[GateFinding] = []
    for rel in TEXT_SURFACES:
        path = repo_root / rel
        if not path.is_file():
            findings.append(GateFinding("disclosure_missing", f"{rel}: file missing"))
        elif DISCLOSURE_LINE not in path.read_text(encoding="utf-8"):
            findings.append(
                GateFinding("disclosure_missing", f"{rel}: disclosure line not found")
            )
    return findings


def _banner_findings(
    repo_root: Path, cards: list[tuple[str, dict[str, Any]]]
) -> list[GateFinding]:
    findings: list[GateFinding] = []
    for dir_name, card in cards:
        if "example" not in provenance_values(card):
            continue
        card_md = repo_root / "cards" / dir_name / "CARD.md"
        text = card_md.read_text(encoding="utf-8") if card_md.is_file() else ""
        if EXAMPLE_BANNER not in text:
            findings.append(
                GateFinding(
                    "example_banner_missing",
                    f"cards/{dir_name}/CARD.md: card carries example-provenance "
                    "records but renders no example banner",
                )
            )
    return findings


def _site_findings(
    site_dist: Path, cards: list[tuple[str, dict[str, Any]]]
) -> list[GateFinding]:
    pages = sorted(site_dist.rglob("*.html"))
    if not pages:
        raise ReleaseGateUsageError(f"{site_dist}: no HTML pages found — not a built site")
    findings: list[GateFinding] = []
    for page in pages:
        if DISCLOSURE_LINE not in page.read_text(encoding="utf-8"):
            findings.append(
                GateFinding(
                    "disclosure_missing",
                    f"{site_dist.name}/{page.relative_to(site_dist).as_posix()}: "
                    "disclosure line not found",
                )
            )
    index_html = site_dist / "index.html"
    any_example = any("example" in provenance_values(card) for _, card in cards)
    if (
        any_example
        and index_html.is_file()
        and SITE_EXAMPLE_MARKER not in index_html.read_text(encoding="utf-8")
    ):
        findings.append(
            GateFinding(
                "example_banner_missing",
                f"{site_dist.name}/index.html: example-provenance records shown "
                "without the fixtures banner",
            )
        )
    return findings


def run_gate(
    repo_root: Path, scope_path: Path, site_dist: Path | None = None
) -> list[GateFinding]:
    """All gate findings, deterministically ordered. Raises
    ReleaseGateUsageError when the gate cannot run at all."""
    scope = load_scope(scope_path)
    try:
        cards = collect_cards(repo_root / "cards")
    except CardIndexError as exc:
        raise ReleaseGateUsageError(f"invalid cards under {repo_root / 'cards'}: {exc}") from exc

    findings = _scope_findings(cards, scope, scope_path.name)
    findings += _disclosure_findings(repo_root)
    findings += _banner_findings(repo_root, cards)
    if site_dist is not None:
        findings += _site_findings(site_dist, cards)
    return findings
