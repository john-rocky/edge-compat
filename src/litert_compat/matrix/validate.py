"""Schema validation plus semantic lint for matrix snapshot files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from litert_compat.matrix.canonical import Signature, format_signature, signature

_REQUIRED_MEASURED_EVIDENCE = ("source_model", "litert_version", "date")

#: NPU backend IDs registered in matrix.schema.json 1.3 (Phase 13). NPU
#: behavior varies per SoC and vendor SDK, so snapshots for these backends
#: must pin their target hardware (semantic lint below).
NPU_BACKENDS = frozenset({"npu_qnn", "npu_neuropilot"})

_REQUIRED_NPU_TARGET = ("device", "soc", "driver")


@dataclass(frozen=True)
class Finding:
    """One validation finding. `code` is stable; `message` is human-readable."""

    code: str
    message: str


class MatrixValidationError(ValueError):
    """Raised when a matrix document fails validation on load."""

    def __init__(self, source: str, findings: list[Finding]) -> None:
        self.findings = findings
        lines = "\n".join(f"  [{f.code}] {f.message}" for f in findings)
        super().__init__(f"invalid matrix document ({source}):\n{lines}")


def load_matrix_schema() -> dict[str, Any]:
    """Load matrix.schema.json — packaged copy first, repo root when running from source."""
    res = resources.files("litert_compat").joinpath("schemas/matrix.schema.json")
    if res.is_file():
        return json.loads(res.read_text(encoding="utf-8"))
    repo_copy = Path(__file__).resolve().parents[3] / "schemas" / "matrix.schema.json"
    return json.loads(repo_copy.read_text(encoding="utf-8"))


def _schema_findings(doc: Any) -> list[Finding]:
    validator = Draft202012Validator(load_matrix_schema())
    errors = sorted(validator.iter_errors(doc), key=lambda e: (list(e.absolute_path), e.message))
    out = []
    for err in errors:
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        out.append(Finding("schema", f"{where}: {err.message}"))
    return out


def _semantic_findings(doc: Any) -> list[Finding]:
    out: list[Finding] = []
    entries = doc.get("entries", []) if isinstance(doc, dict) else []
    by_signature: dict[Signature, list[tuple[int, str]]] = defaultdict(list)

    if isinstance(doc, dict) and doc.get("backend") in NPU_BACKENDS:
        target = doc.get("target") or {}
        missing = [k for k in _REQUIRED_NPU_TARGET if not target.get(k)]
        if missing:
            out.append(
                Finding(
                    "npu_target_required",
                    f"backend {doc['backend']!r} is an NPU backend: NPU behavior varies "
                    f"per SoC and vendor SDK, so target must carry "
                    f"{', '.join(_REQUIRED_NPU_TARGET)} (missing: {', '.join(missing)})",
                )
            )

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if entry.get("provenance") == "measured":
            evidence = entry.get("evidence") or {}
            missing = [k for k in _REQUIRED_MEASURED_EVIDENCE if not evidence.get(k)]
            if missing:
                out.append(
                    Finding(
                        "missing_evidence",
                        f"entries[{i}] ({entry.get('op')}): measured entry missing "
                        f"evidence field(s): {', '.join(missing)}",
                    )
                )
        by_signature[signature(entry)].append((i, entry.get("status", "")))

    for sig, occurrences in sorted(by_signature.items()):
        statuses = {status for _, status in occurrences}
        if len(statuses) > 1:
            detail = ", ".join(f"entries[{i}]={status}" for i, status in occurrences)
            out.append(
                Finding(
                    "contradictory_entries",
                    f"contradictory verdicts for signature {format_signature(sig)}: {detail}",
                )
            )
    return out


def validate_document(doc: Any) -> list[Finding]:
    """Full validation: JSON Schema first, then semantic lint. Empty list = pass."""
    return _schema_findings(doc) + _semantic_findings(doc)


def provenance_counts(doc: Any) -> dict[str, int]:
    """Entries per provenance value. The `inferred` count is the staleness debt."""
    counts: Counter[str] = Counter()
    if isinstance(doc, dict):
        for entry in doc.get("entries", []):
            if isinstance(entry, dict):
                counts[str(entry.get("provenance"))] += 1
    return dict(counts)
