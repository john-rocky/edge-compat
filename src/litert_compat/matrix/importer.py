"""CSV import: the neutral interchange format for the owner's curated op table.

Forgiving on input (whitespace trimmed, any column order, unknown columns
warned and ignored), strict on output (the result must pass validation).
Deterministic: the same CSV and options produce byte-identical JSON.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litert_compat.matrix.canonical import sort_entries

KNOWN_COLUMNS = frozenset(
    {
        "op",
        "dtypes",
        "constraints",
        "status",
        "conditions",
        "rewrite_symptom",
        "rewrite",
        "rewrite_expected_effect",
        "provenance",
        "evidence_source_model",
        "evidence_litert_version",
        "evidence_date",
    }
)

_STATUS_VALUES = frozenset({"delegated", "partial", "fallback", "incorrect", "crash", "unknown"})
_PROVENANCE_VALUES = frozenset({"measured", "vendor_doc", "inferred", "example"})


class CsvImportError(ValueError):
    """CSV rows could not be converted to valid matrix entries."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("CSV import failed:\n" + "\n".join(f"  {e}" for e in errors))


@dataclass
class ImportResult:
    doc: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _parse_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_constraints(cell: str, line: int, errors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in cell.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            errors.append(f"line {line}: constraint '{part}' is not key=value")
            continue
        key, _, raw = part.partition("=")
        out[key.strip()] = _parse_scalar(raw.strip())
    return out


def _row_entry(row: dict[str, str], line: int, errors: list[str]) -> dict[str, Any] | None:
    get = lambda col: (row.get(col) or "").strip()  # noqa: E731

    op = get("op")
    status = get("status")
    provenance = get("provenance")
    row_errors: list[str] = []
    if not op:
        row_errors.append(f"line {line}: missing required column 'op'")
    if status not in _STATUS_VALUES:
        row_errors.append(
            f"line {line}: status '{status}' not one of {sorted(_STATUS_VALUES)}"
        )
    if provenance not in _PROVENANCE_VALUES:
        row_errors.append(
            f"line {line}: provenance '{provenance}' not one of {sorted(_PROVENANCE_VALUES)}"
        )
    if row_errors:
        errors.extend(row_errors)
        return None

    entry: dict[str, Any] = {"op": op, "status": status, "provenance": provenance}

    dtypes = sorted({d.strip() for d in get("dtypes").split(";") if d.strip()})
    if dtypes:
        entry["dtypes"] = dtypes

    constraints = _parse_constraints(get("constraints"), line, errors)
    if constraints:
        entry["constraints"] = constraints

    if get("conditions"):
        entry["conditions"] = get("conditions")

    symptom, rewrite = get("rewrite_symptom"), get("rewrite")
    effect = get("rewrite_expected_effect")
    if symptom or rewrite or effect:
        if not (symptom and rewrite):
            errors.append(
                f"line {line}: a rewrite hint needs both rewrite_symptom and rewrite"
            )
        else:
            hint = {"symptom": symptom, "rewrite": rewrite}
            if effect:
                hint["expected_effect"] = effect
            entry["rewrite_hints"] = [hint]

    evidence = {
        key: get(f"evidence_{key}")
        for key in ("source_model", "litert_version", "date")
        if get(f"evidence_{key}")
    }
    if evidence:
        entry["evidence"] = evidence

    return entry


def import_csv(
    csv_path: Path,
    *,
    litert_version: str,
    backend: str,
    generated_at: str,
    target: dict[str, str] | None = None,
) -> ImportResult:
    warnings: list[str] = []
    errors: list[str] = []
    entries: list[dict[str, Any]] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header_row = next(reader)
        except StopIteration:
            raise CsvImportError(["CSV file is empty"]) from None
        header = [h.strip().lower().replace(" ", "_") for h in header_row]
        unknown = [h for h in header if h and h not in KNOWN_COLUMNS]
        if unknown:
            warnings.append(f"ignoring unknown column(s): {', '.join(unknown)}")

        for line, raw in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw):
                continue
            row = {col: value for col, value in zip(header, raw, strict=False) if col}
            entry = _row_entry(row, line, errors)
            if entry is not None:
                entries.append(entry)

    if errors:
        raise CsvImportError(errors)

    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "litert_version": litert_version,
        "backend": backend,
        "generated_at": generated_at,
        "entries": entries,
    }
    if target:
        doc["target"] = {k: v for k, v in target.items() if v}
    sort_entries(doc)
    return ImportResult(doc=doc, warnings=warnings)
