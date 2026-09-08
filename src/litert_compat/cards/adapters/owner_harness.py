"""Adapter stub: the owner's benchmark harness exports -> benchmark_result.schema.json.

The mapping depends on sample exports that have not been provided yet (see
PROGRESS.md "Ask the owner"). Per the data integrity rules, the mapping is NOT
guessed: every function below raises NotImplementedError until the owner
supplies one sample export per source and confirms the field mapping.

Known sources this adapter must eventually cover (spec, Integration Rules §E
Phase 3 — the schema was designed so these map on without schema changes):

- devicemark leaderboard JSONL (`~/code/devicemark/data/leaderboard/measurements.jsonl`):
  rows like {artifact_id, runtime, device, decode_tok_s, peak_mem_mb,
  mem_measured, power_w}. Expected mapping: artifact_id -> model_id,
  decode_tok_s -> tokens_per_s, peak_mem_mb -> peak_mem_mb,
  power_w -> metrics.power_w — TODO(owner): confirm on a sample export,
  including harness/harness_version/date, which the rows do not carry.
- apple-silicon-llm-bench exports — TODO(owner): provide one sample export.
- gpu_audit outputs (`litertlm-convert/reports/gpu_audit/`) — TODO(owner):
  provide one sample log; note these carry PASS/FAIL + unsupported-op data,
  which may belong in the matrix rather than in benchmark records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def convert_devicemark_jsonl(path: Path, model_id: str) -> dict[str, Any]:
    """devicemark measurements.jsonl rows for `model_id` -> a benchmark results file dict."""
    raise NotImplementedError(
        "TODO(owner): confirm the devicemark field mapping on a sample export "
        "(see module docstring); do not guess harness/date fields"
    )


def convert_owner_harness_export(path: Path) -> dict[str, Any]:
    """The owner's cross-runtime bench harness export -> a benchmark results file dict."""
    raise NotImplementedError(
        "TODO(owner): provide one sample harness export to fix the field mapping"
    )
