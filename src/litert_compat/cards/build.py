"""Assemble card.json from validated inputs.

The card formats and links measured data; it never produces measurements.
Benchmark records are carried verbatim from the ingested results file, the
`delegation` block is mapped directly from `edge-lint --json` output, and
cross-input identity is enforced: the lint report must describe the exact model
file (sha256), and benchmark files must name the card's model id.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from litert_compat.cards.schema_io import schema_errors

CARD_SCHEMA_VERSION = "1.0"


class CardBuildError(ValueError):
    """Card inputs are unusable; `errors` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        lines = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"cannot build card:\n{lines}")


def _delegation_from_lint(report: dict[str, Any]) -> dict[str, Any]:
    """Map lint_report.schema.json output to the card's delegation block."""
    blocking_ops: list[str] = []
    for b in report["blocking_ops"]:  # ranking order is deterministic; dedup keeps first
        if b["op"] not in blocking_ops:
            blocking_ops.append(b["op"])
    return {
        "backend": report["backend"],
        "litert_version": report["litert_version"],
        "coverage_ops_pct": report["summary"]["coverage_ops_pct"],
        "partitions": report["summary"]["partition_count"],
        "blocking_ops": blocking_ops,
        "matched_provenance_counts": report["summary"]["matched_provenance_counts"],
        "lint_report_version": report["schema_version"],
    }


def build_card(
    model_path: Path | None,
    meta: dict[str, Any],
    bench: dict[str, Any] | None = None,
    lint_report: dict[str, Any] | None = None,
    cross_bench: dict[str, Any] | None = None,
    remote_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a card.schema.json-valid card object. Raises CardBuildError on defects.

    Artifact identity comes from exactly one place: `model_path` (bytes are
    hashed locally) or `remote_artifact` — ``{"file", "sha256", "size_bytes"}``
    taken verbatim from the hosting API (e.g. the HF LFS entry) for artifacts
    whose bytes are published but not on this disk. Never hand-compute or
    guess a remote hash; copy it from the host's metadata.
    """
    errors: list[str] = []
    model_id = meta["model"]["id"]
    if (model_path is None) == (remote_artifact is None):
        raise CardBuildError(
            ["exactly one of model_path / remote_artifact must be given"]
        )
    if model_path is not None:
        model_bytes = model_path.read_bytes()
        sha256 = hashlib.sha256(model_bytes).hexdigest()
        artifact_file = model_path.name
        size_bytes = len(model_bytes)
    else:
        assert remote_artifact is not None
        missing = [k for k in ("file", "sha256", "size_bytes") if not remote_artifact.get(k)]
        if missing:
            raise CardBuildError([f"remote_artifact is missing {', '.join(missing)}"])
        sha256 = str(remote_artifact["sha256"]).lower()
        artifact_file = str(remote_artifact["file"])
        size_bytes = int(remote_artifact["size_bytes"])

    for label, doc in (("bench", bench), ("cross-bench", cross_bench)):
        if doc is not None and doc["model_id"] != model_id:
            errors.append(
                f"{label} file is for model {doc['model_id']!r}, "
                f"but meta model.id is {model_id!r}"
            )
    if cross_bench is not None:
        for i, record in enumerate(cross_bench["results"]):
            if not record.get("runtime"):
                errors.append(f"cross-bench results[{i}]: `runtime` is required on every record")
    if lint_report is not None and lint_report["model"]["sha256"] != sha256:
        errors.append(
            f"lint report describes a different model: report sha256 "
            f"{lint_report['model']['sha256'][:12]}… != {artifact_file} sha256 {sha256[:12]}…"
        )
    if errors:
        raise CardBuildError(errors)

    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "model": meta["model"],
        "conversion": meta["conversion"],
        "artifacts": [
            {
                "file": artifact_file,
                "sha256": sha256,
                "size_mb": round(size_bytes / (1024 * 1024), 3),
            }
        ],
        "benchmarks": bench["results"] if bench is not None else [],
        "delegation": _delegation_from_lint(lint_report) if lint_report is not None else None,
        "cross_runtime": cross_bench["results"] if cross_bench is not None else [],
        "pitfalls": meta["pitfalls"],
    }

    result_errors = schema_errors(card, "card.schema.json")
    if result_errors:  # inputs validated, so this indicates a generator bug — still never write
        raise CardBuildError([f"built card is schema-invalid: {e}" for e in result_errors])
    return card
