"""Adapter stub: benchmark harness exports -> benchmark_result.schema.json.

The mapping depends on sample exports that have not been provided yet (see
an open question). Per the data integrity rules, the mapping is NOT
guessed: every function below raises NotImplementedError until someone
supplies one sample export per source and confirms the field mapping.

Known sources this adapter must eventually cover (the schema was
designed so these map on without schema changes):

- devicemark leaderboard JSONL (`data/leaderboard/measurements.jsonl` of the devicemark project):
  rows like {artifact_id, runtime, device, decode_tok_s, peak_mem_mb,
  mem_measured, power_w}. Expected mapping: artifact_id -> model_id,
  decode_tok_s -> tokens_per_s, peak_mem_mb -> peak_mem_mb,
  power_w -> metrics.power_w — TODO: confirm on a sample export,
  including harness/harness_version/date, which the rows do not carry.
- apple-silicon-llm-bench exports — TODO: needs one sample export.
- gpu_audit outputs (the gpu_audit reports) — TODO:
  provide one sample log; note these carry PASS/FAIL + unsupported-op data,
  which may belong in the matrix rather than in benchmark records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def convert_devicemark_jsonl(path: Path, model_id: str) -> dict[str, Any]:
    """devicemark measurements.jsonl rows for `model_id` -> a benchmark results file dict."""
    raise NotImplementedError(
        "TODO: confirm the devicemark field mapping on a sample export "
        "(see module docstring); do not guess harness/date fields"
    )


def convert_owner_harness_export(path: Path) -> dict[str, Any]:
    """A cross-runtime bench harness export -> a benchmark results file dict."""
    raise NotImplementedError(
        "TODO: one sample harness export is needed to fix the field mapping"
    )
