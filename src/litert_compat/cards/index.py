"""Deterministic catalog outputs: cards/index.json, cards/README.md, llms.txt.

llms.txt is the single entry point from which AI agents and crawlers can
discover the whole dataset: every CARD.md plus the matrix snapshots under
data/matrix/. All three outputs are byte-identical across reruns on the same
inputs (sorted iteration, canonical JSON).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litert_compat.branding import DISCLOSURE_LINE, NAMING_NOTICE, PROJECT_NAME
from litert_compat.cards.schema_io import schema_errors
from litert_compat.device_runs.records import device_status, record_label
from litert_compat.matrix.canonical import load_json

INDEX_SCHEMA_VERSION = "1.2"

# One-line agent prompt (Phase 12.2): rebuilds a card from the committed
# example inputs with real commands only, mirroring what build-all does for
# the first manifest row (the committed card additionally carries the device
# block that `edge-card enrich --device-runs` merges).
CARDS_REPRODUCE_PROMPT = (
    f"In the {PROJECT_NAME} repo, rebuild a model card from the committed example "
    "inputs: run `uv run edge-card build "
    "--model data/examples/model_clean_example.tflite "
    "--meta data/examples/meta_clean_example.yaml "
    "--bench data/examples/bench_clean_example.json "
    "--lint data/examples/lint_clean_example.json "
    "--cross-bench data/examples/cross_runtime_clean_example.json "
    "-o /tmp/example-card` and compare the output with `cards/example-tiny-clean/` "
    "(the committed card additionally carries the device block that "
    "`edge-card enrich --device-runs` merges)."
)


def browser_status(record: dict[str, Any]) -> str:
    """Deterministic one-word summary of one browser backend record.

    `pass` claims only what the record shows: loaded, ran, and neither an
    output mismatch nor a delegation fallback was recorded. The precise
    failure class and diff values live in the card's browser block and the
    linked sweep_source file — this is a table cell, not the evidence.
    """
    if not record["loads"]:
        return "load_failed"
    if not record["runs"]:
        return "run_failed"
    if record["output_match"] is False:
        return "output_mismatch"
    if record["full_delegation"] is False:
        return "fallback"
    return "pass"


class CardIndexError(ValueError):
    """One or more card.json files are invalid; `errors` lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        lines = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"cannot index cards:\n{lines}")


def collect_cards(cards_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load every cards/<id>/card.json, sorted by directory name. Raises on invalid cards."""
    errors: list[str] = []
    cards: list[tuple[str, dict[str, Any]]] = []
    for card_path in sorted(cards_dir.glob("*/card.json")):
        try:
            card = load_json(card_path)
        except ValueError as exc:
            errors.append(f"{card_path}: not valid JSON: {exc}")
            continue
        card_errors = schema_errors(card, "card.schema.json")
        if card_errors:
            errors.extend(f"{card_path}: {e}" for e in card_errors)
            continue
        dir_name = card_path.parent.name
        if card["model"]["id"] != dir_name:
            errors.append(
                f"{card_path}: model.id {card['model']['id']!r} != directory name {dir_name!r}"
            )
            continue
        cards.append((dir_name, card))
    if errors:
        raise CardIndexError(errors)
    return cards


def _index_device(device: dict[str, Any] | None) -> dict[str, Any] | None:
    """The index's per-model device field: one status row per device record
    — keyed by `label` (accelerator plus the prompt-length condition, e.g.
    'gpu@13tok', DECISIONS #162), since a device x accelerator pair can carry
    one row per condition — with the env facts the site needs for stale
    flags, plus the snapshot files behind them."""
    if device is None:
        return None
    return {
        "records": [
            {
                "device": entry["device"],
                "accelerator": entry["run"]["accelerator"],
                "label": record_label(entry["run"]),
                "status": device_status(entry["run"]),
                "runtime": entry["run"]["env"]["runtime"],
                "runtime_version": entry["run"]["env"]["runtime_version"],
                "date": entry["run"]["date"],
            }
            for entry in device["records"]
        ],
        "sources": sorted({entry["source"] for entry in device["records"]}),
    }


def build_index_doc(cards: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    models = []
    for dir_name, card in cards:
        delegation = card["delegation"]
        browser = card.get("browser")
        models.append(
            {
                "id": card["model"]["id"],
                "family": card["model"]["family"],
                "task": card["model"]["task"],
                "license": card["model"]["license"],
                "card_json": f"cards/{dir_name}/card.json",
                "card_md": f"cards/{dir_name}/CARD.md",
                "delegation": None
                if delegation is None
                else {
                    "backend": delegation["backend"],
                    "litert_version": delegation["litert_version"],
                    "coverage_ops_pct": delegation["coverage_ops_pct"],
                    "partitions": delegation["partitions"],
                },
                "browser": None
                if browser is None
                else {
                    "statuses": {
                        r["backend"]: browser_status(r) for r in browser["backends"]
                    },
                    "sweep_source": browser["sweep_source"],
                },
                "device": _index_device(card.get("device")),
                "benchmark_count": len(card["benchmarks"]),
                "cross_runtime_count": len(card["cross_runtime"]),
                "pitfall_count": len(card["pitfalls"]),
            }
        )
    return {"schema_version": INDEX_SCHEMA_VERSION, "models": models}


def render_index_markdown(index_doc: dict[str, Any]) -> str:
    lines = [
        "# Model cards",
        "",
        DISCLOSURE_LINE,
        NAMING_NOTICE,
        "",
        "Generated by `edge-card index` — do not edit by hand. One row per model;",
        "`card.json` is the machine-readable source of truth, `CARD.md` the rendered view.",
        "",
        "| Model | Task | Family | Delegation | Browser | Device | Benchmarks | Card |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in index_doc["models"]:
        d = m["delegation"]
        delegation_cell = (
            f"{d['backend']} @ litert {d['litert_version']}: "
            f"{d['coverage_ops_pct']}% · {d['partitions']} partition(s)"
            if d
            else "-"
        )
        b = m["browser"]
        browser_cell = (
            " · ".join(f"{backend}: {status}" for backend, status in sorted(b["statuses"].items()))
            if b
            else "-"
        )
        dev = m["device"]
        device_cell = (
            " · ".join(
                f"{r['device']} {r['label']}: {r['status']}" for r in dev["records"]
            )
            if dev
            else "-"
        )
        lines.append(
            f"| {m['id']} | {m['task']} | {m['family']} | {delegation_cell} | {browser_cell} "
            f"| {device_cell} | {m['benchmark_count']} | [CARD.md]({m['id']}/CARD.md) |"
        )
    lines += [
        "",
        "## Reproduce this with an agent",
        "",
        "One-line prompt for Claude Code / Gemini CLI (real commands only):",
        "",
        f"> {CARDS_REPRODUCE_PROMPT}",
    ]
    return "\n".join(lines) + "\n"


def render_llms_txt(index_doc: dict[str, Any], matrix_files: list[str]) -> str:
    lines = [
        f"# {PROJECT_NAME}",
        "",
        "> Delegate compatibility data for LiteRT: an op-level compatibility matrix "
        "per (backend x LiteRT version), "
        "plus per-model cards with measured benchmarks and static delegation verdicts. "
        "Machine consumption first: stable schemas under schemas/, deterministic JSON, "
        "exact TFLite builtin op names. Paths are repo-relative.",
        "",
        DISCLOSURE_LINE,
        NAMING_NOTICE,
        "",
        "## Model cards",
        "",
    ]
    if index_doc["models"]:
        for m in index_doc["models"]:
            lines.append(
                f"- [{m['id']}]({m['card_md']}): {m['task']} — machine-readable "
                f"source of truth at {m['card_json']}"
            )
    else:
        lines.append("- (none yet)")
    lines += ["", "## Compatibility matrix snapshots", ""]
    if matrix_files:
        lines.extend(
            f"- [{Path(f).stem}]({f}): op-level delegate support snapshot" for f in matrix_files
        )
    else:
        lines.append("- (none yet — see PROGRESS.md; example data under data/examples/)")
    lines += ["", "## Browser sweep results", ""]
    sweep_sources = sorted(
        {m["browser"]["sweep_source"] for m in index_doc["models"] if m["browser"] is not None}
    )
    if sweep_sources:
        lines.extend(
            f"- [{Path(f).stem}]({f}): model-level LiteRT.js sweep record "
            "(sweep_result.schema.json)"
            for f in sweep_sources
        )
    else:
        lines.append("- (none yet — cards gain browser blocks via `edge-card enrich`)")
    lines += ["", "## Device-run results", ""]
    device_sources = sorted(
        {
            source
            for m in index_doc["models"]
            if m["device"] is not None
            for source in m["device"]["sources"]
        }
    )
    if device_sources:
        lines.extend(
            f"- [{Path(f).stem}]({f}): model-level device-run record — NPU / LiteRT-LM "
            "lanes (device_run_result.schema.json)"
            for f in device_sources
        )
    else:
        lines.append(
            "- (none yet — cards gain device blocks via `edge-card enrich --device-runs`)"
        )
    return "\n".join(lines) + "\n"


def matrix_snapshot_files(repo_root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(repo_root)) for p in (repo_root / "data" / "matrix").glob("*.json")
    )
