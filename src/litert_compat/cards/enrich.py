"""Merge browser sweep results into cards (`edge-card enrich`, Phase 6).

Model-level enrichment ONLY — the trap rule in full force: this module never
creates, edits, or infers matrix entries. "Model X fully delegated" licenses no
per-op verdict, and a fallback attributes nothing to any specific op; op-level
web entries require a runtime log naming the op, captured as evidence, and are
out of scope here.

Browser block fields map 1:1 from the sweep record (backend, loads, runs,
full_delegation, output_match, max_rel_diff, latency_p50_ms, env, date,
provenance); the full record — failure_class, error, delegation_evidence,
sweep config — stays in the sweep file, linked via `sweep_source`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litert_compat.cards.schema_io import schema_errors
from litert_compat.device_runs.records import (
    device_run_errors,
    discover_device_run_snapshots,
    record_label,
    record_sort_key,
    select_device_runs,
)
from litert_compat.matrix.canonical import load_json

ENRICHED_CARD_SCHEMA_VERSION = "1.1"
DEVICE_ENRICHED_CARD_SCHEMA_VERSION = "1.2"

_RECORD_FIELDS = (
    "backend",
    "loads",
    "runs",
    "full_delegation",
    "output_match",
    "max_rel_diff",
    "latency_p50_ms",
    "env",
    "date",
    "provenance",
)


class SweepLoadError(ValueError):
    """One or more sweep result files are invalid; `errors` lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        lines = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"cannot load sweep results:\n{lines}")


class CardEnrichError(ValueError):
    """The enriched card failed schema validation (a generator bug — never write it)."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        lines = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"cannot enrich card:\n{lines}")


def load_sweep_results(sweep_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load every <sweep_dir>/*.json, sorted by filename, validated against
    sweep_result.schema.json. Raises SweepLoadError listing every defect."""
    errors: list[str] = []
    results: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(sweep_dir.glob("*.json")):
        try:
            doc = load_json(path)
        except ValueError as exc:
            errors.append(f"{path}: not valid JSON: {exc}")
            continue
        doc_errors = schema_errors(doc, "sweep_result.schema.json")
        if doc_errors:
            errors.extend(f"{path}: {e}" for e in doc_errors)
            continue
        if doc["model_id"] != path.stem:
            errors.append(f"{path}: model_id {doc['model_id']!r} != file basename {path.stem!r}")
            continue
        results.append((path, doc))
    if errors:
        raise SweepLoadError(errors)
    return results


def repo_relative(path: Path, repo_root: Path) -> str:
    """POSIX path of `path` relative to the repo root; as given when outside it."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _enriched_version(card: dict[str, Any]) -> str:
    """A card carrying `device` declares 1.2; browser alone declares 1.1."""
    if "device" in card:
        return DEVICE_ENRICHED_CARD_SCHEMA_VERSION
    return ENRICHED_CARD_SCHEMA_VERSION


def enrich_card(
    card: dict[str, Any], sweep: dict[str, Any], sweep_source: str
) -> dict[str, Any]:
    """-> a copy of `card` with its browser block replaced from `sweep`.

    An existing non-null demo_url survives re-enrichment (Phase 7 fills it);
    everything else in the browser block is authored by the sweep file alone.
    """
    existing = card.get("browser")
    enriched = dict(card)
    enriched["browser"] = {
        "backends": [{k: record[k] for k in _RECORD_FIELDS} for record in sweep["results"]],
        "demo_url": existing["demo_url"] if existing else None,
        "sweep_source": sweep_source,
    }
    enriched["schema_version"] = _enriched_version(enriched)
    errors = schema_errors(enriched, "card.schema.json")
    if errors:  # inputs validated, so this indicates a generator bug — still never write
        raise CardEnrichError([f"enriched card is schema-invalid: {e}" for e in errors])
    return enriched


def load_device_runs(dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Load every *.json in the given snapshot directories, sorted by file
    name per directory, validated against device_run_result.schema.json.
    Raises SweepLoadError listing every defect (same collected-error shape
    as the sweep loader)."""
    errors: list[str] = []
    results: list[tuple[Path, dict[str, Any]]] = []
    seen: dict[tuple[str, str, str], Path] = {}
    for directory in dirs:
        for path in sorted(directory.glob("*.json")):
            try:
                doc = load_json(path)
            except ValueError as exc:
                errors.append(f"{path}: not valid JSON: {exc}")
                continue
            doc_errors = device_run_errors(doc)
            if doc_errors:
                errors.extend(f"{path}: {e}" for e in doc_errors)
                continue
            expected = f"{doc['model_id']}__{doc['device']}.json"
            if path.name != expected:
                errors.append(f"{path}: file name must be {expected}")
                continue
            for record in doc["results"]:
                key = (doc["model_id"], doc["device"], record_label(record))
                if key in seen:
                    errors.append(
                        f"{path}: record for {'/'.join(key)} already loaded from "
                        f"{seen[key]} — pass one snapshot per (device x accelerator "
                        "x prompt-length condition)"
                    )
                else:
                    seen[key] = path
            results.append((path, doc))
    if errors:
        raise SweepLoadError(errors)
    return results


def enrich_card_device(
    card: dict[str, Any], runs: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """-> a schema-1.2 copy of `card` with its device block replaced from the
    given (repo-relative source, device-run document) pairs. Accelerator
    records are embedded verbatim — model-level facts only, never matrix
    entries (the trap rule, device edition)."""
    entries = [
        {"device": doc["device"], "source": source, "run": record}
        for source, doc in runs
        for record in doc["results"]
    ]
    enriched = dict(card)
    enriched["device"] = {
        "records": sorted(entries, key=lambda e: (e["device"], record_sort_key(e["run"])))
    }
    enriched["schema_version"] = _enriched_version(enriched)
    errors = schema_errors(enriched, "card.schema.json")
    if errors:  # inputs validated, so this indicates a generator bug — still never write
        raise CardEnrichError([f"enriched card is schema-invalid: {e}" for e in errors])
    return enriched


def _device_cells(card: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """(device, record_label) -> entry, for every device record a card carries."""
    device = card.get("device")
    if device is None:
        return {}
    return {(e["device"], record_label(e["run"])): e for e in device["records"]}


def device_cells_dropped(
    card: dict[str, Any], enriched: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """Cells the card carries that `enriched` would not, as sorted (device,
    label, source) — the replace-trap check (DECISIONS #161/#165). `enrich`
    rebuilds a device block from exactly the snapshots it was given, so a
    snapshot left off the command line removes the rows it contributed with
    no signal but the diff; the CLI refuses to write such a card unless the
    caller states the drop."""
    after = _device_cells(enriched)
    return sorted(
        (device, label, entry["source"])
        for (device, label), entry in _device_cells(card).items()
        if (device, label) not in after
    )


def device_coverage_findings(
    cards: list[tuple[str, dict[str, Any]]], roots: list[Path], repo_root: Path
) -> list[str]:
    """Every way the cards' device blocks differ from the cell-wise newest
    selection over the snapshot roots (DECISIONS #165): a measured cell a
    card lacks, a card cell a newer snapshot supersedes, a card record that
    no longer matches the file it cites, a cited file that does not exist.
    Card rows citing a file outside `roots` are checked for existence only —
    a root that was not passed cannot judge them. Sorted; empty means the
    cards are exactly what enriching from the roots would write."""
    snapshots = [s for root in roots for s in discover_device_run_snapshots(root)]
    expected: dict[str, dict[tuple[str, str], tuple[str, dict[str, Any]]]] = {}
    for path, doc in select_device_runs(snapshots):
        source = repo_relative(path, repo_root)
        cells = expected.setdefault(doc["model_id"], {})
        for record in doc["results"]:
            cells[(doc["device"], record_label(record))] = (source, record)
    prefixes = tuple(repo_relative(root, repo_root).rstrip("/") + "/" for root in roots)
    findings: list[str] = []
    for dir_name, card in cards:
        want = expected.get(card["model"]["id"], {})
        have = _device_cells(card)
        for (device, label), entry in have.items():
            cell = f"{dir_name}: {device} {label}"
            source = entry["source"]
            if not (repo_root / source).is_file():
                findings.append(f"{cell} cites {source}, which does not exist")
            elif not source.startswith(prefixes):
                continue
            elif (device, label) not in want:
                findings.append(
                    f"{cell} cites {source}, which no longer measures that cell — re-enrich"
                )
            elif want[(device, label)][0] != source:
                findings.append(
                    f"{cell} is from {source}; the newest measurement is "
                    f"{want[(device, label)][0]} — re-enrich"
                )
            elif want[(device, label)][1] != entry["run"]:
                findings.append(f"{cell} differs from its source {source} — re-enrich")
        for (device, label), (source, _record) in want.items():
            if (device, label) not in have:
                findings.append(
                    f"{dir_name}: {device} {label} measured in {source} is not on the card "
                    "— enrich dropped it or never attached it"
                )
    return sorted(findings)
