"""Shared factories for building matrix documents and device-run snapshots in
tests (example provenance only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litert_compat.device_runs.adapters import make_env
from litert_compat.matrix.canonical import canonical_dumps


def make_doc(entries: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "litert_version": "0.0.0-example",
        "backend": "gpu_mldrift",
        "generated_at": "2026-08-10",
        "entries": entries,
    }
    doc.update(overrides)
    return doc


def entry(
    op: str = "CONV_2D",
    status: str = "delegated",
    provenance: str = "example",
    **fields: Any,
) -> dict[str, Any]:
    e: dict[str, Any] = {"op": op, "status": status, "provenance": provenance}
    e.update(fields)
    return e


def example_record(
    accelerator: str = "gpu",
    *,
    runtime: str = "litert-lm",
    runtime_version: str = "0.0.0-example",
    loads: bool = True,
    runs: bool = True,
    output_match: bool | None = None,
    full_delegation: bool | None = None,
    decode: float | None = 40.0,
    latency: float | None = None,
    failure_class: str | None = None,
    date: str = "2026-01-01",
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "accelerator": accelerator,
        "loads": loads,
        "runs": runs,
        "failure_class": failure_class,
        "error": None,
        "full_delegation": full_delegation,
        "delegated_ops": None,
        "total_ops": None,
        "output_match": output_match,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "latency_p50_ms": latency,
        "prefill_tokens_per_s": None,
        "decode_tokens_per_s": decode,
        "ttft_ms": None,
        "peak_mem_mb": None,
        "context_length": None,
        "evidence": [],
        "env": make_env(
            device="Example Device", runtime=runtime, runtime_version=runtime_version
        ),
        "date": date,
        "provenance": "example",
    }
    if metrics is not None:
        record["metrics"] = metrics
    return record


def example_doc(
    model_id: str, device: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "device": device,
        "artifact": f"{model_id}.litertlm",
        "quantization": None,
        "results": records,
    }


def write_snapshot(
    root: Path, version: str, date: str, docs: list[dict[str, Any]]
) -> Path:
    snap = root / version / date
    snap.mkdir(parents=True)
    for doc in docs:
        path = snap / f"{doc['model_id']}__{doc['device']}.json"
        path.write_text(canonical_dumps(doc), encoding="utf-8")
    return snap
