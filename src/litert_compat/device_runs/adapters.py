"""Ingestion adapters: existing measurement-tool outputs -> device-run records.

Three lanes, mapped against the REAL samples staged in
`data/examples/llm_samples/` (never guessed — the Phase 3 adapter rule), with
the owner decisions recorded in that directory's README encoded here:

- gpu_audit (`gpu_gate_mac.sh` logs): the verdict comes from the LOG BODY
  only — the CLI exits 0 even when engine creation fails, and `summary.txt`
  mixes historical line formats and is not 1:1 with surviving logs, so
  neither is an adapter input. A printed results block with zero throughput
  is a FAILURE (`zero_throughput`), never a measurement. The 2026-07-22
  batch's speed figures are import-blocked (`speeds_contaminated=True`)
  while their PASS/FAIL verdicts remain usable (owner decision #5).
- devicemark leaderboard rows: only `*__litertlm` artifacts are this lane
  (`coreai` rows belong in a card's cross_runtime list); `mem_measured:
  false` memory is carried as the explicitly-estimated
  `metrics.peak_mem_est_mb`, never as a measured `peak_mem_mb` (owner
  decision #4).
- compat_check reports: refuse-until-sampled (owner decision #3) — only the
  `ok` status branch has a staged real sample, so any other status (and any
  lane other than `llm`) is refused with a precise reason instead of built
  against invented strings.

Fields a source does not carry (dates, runtime versions, host hardware) are
supplied by the caller or absent — never inferred from file names or mtimes.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field, replace
from typing import Any

from litert_compat.matrix.canonical import canonical_dumps

#: Evidence lines keep the diagnostic head verbatim; graph-location traces
#: beyond this length are cut with an explicit marker (honest truncation).
_EVIDENCE_LINE_CAP = 200

_UNSUPPORTED_HEADER = "ERROR: Following operations are not supported by GPU delegate:"
_PARTITION_RE = re.compile(
    r"^(\d+) operations will run on the GPU, "
    r"and the remaining (\d+) operations will run on the CPU\.$"
)
_OP_LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]*): (.*)$")

CONTAMINATED_NOTE = (
    "speed figures withheld: this measurement batch was retracted as contaminated "
    "(parallel GPU load; owner correction in gpu_audit MATRIX.md, 2026-07-23) — "
    "the PASS/FAIL verdict is load-independent and stands"
)


class AdapterError(ValueError):
    """The source cannot be ingested honestly; the message says exactly why."""


def _truncate(line: str) -> str:
    if len(line) <= _EVIDENCE_LINE_CAP:
        return line
    return line[:_EVIDENCE_LINE_CAP] + " …[trace truncated]"


def make_env(
    *,
    device: str,
    runtime: str,
    runtime_version: str,
    soc: str | None = None,
    vendor_sdk: str | None = None,
    os_build: str | None = None,
    machine_label: str | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "device": device,
        "soc": soc,
        "runtime": runtime,
        "runtime_version": runtime_version,
        "vendor_sdk": vendor_sdk,
        "os_build": os_build,
    }
    if machine_label is not None:
        env["machine_label"] = machine_label
    return env


def _base_record(accelerator: str, env: dict[str, Any], date: str) -> dict[str, Any]:
    """All nullable measurement fields start as explicit nulls: a field is
    filled only when the source states it."""
    return {
        "accelerator": accelerator,
        "loads": False,
        "runs": False,
        "failure_class": None,
        "error": None,
        "full_delegation": None,
        "delegated_ops": None,
        "total_ops": None,
        "output_match": None,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "latency_p50_ms": None,
        "prefill_tokens_per_s": None,
        "decode_tokens_per_s": None,
        "ttft_ms": None,
        "peak_mem_mb": None,
        "context_length": None,
        "evidence": [],
        "env": env,
        "date": date,
        "provenance": "measured",
    }


# --- lane 1: gpu_audit -------------------------------------------------------


@dataclass
class GpuAuditLog:
    """Parsed `gpu_gate_mac.sh` log. Three shapes exist and all are staged:
    PASS with a results block, FAIL before engine creation (unsupported-op
    block), and ran-but-zero-throughput (a results block full of 0.00)."""

    artifact: str
    accelerator: str
    prefill_tokens: int | None = None
    decode_tokens: int | None = None
    max_num_tokens: int | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    init_s: float | None = None
    ttft_s: float | None = None
    has_results: bool = False
    engine_failed: bool = False
    engine_error: str | None = None
    unsupported_ops: list[str] = field(default_factory=list)
    unsupported_lines: list[str] = field(default_factory=list)
    partition_line: str | None = None
    delegated_ops: int | None = None
    total_ops: int | None = None
    generate_failures: list[str] = field(default_factory=list)


def _header_int(line: str, prefix: str) -> int | None:
    if not line.startswith(prefix):
        return None
    match = re.search(r":\s*(\d+)\s*$", line)
    return int(match.group(1)) if match else None


def _results_float(line: str, prefix: str) -> float | None:
    if not line.startswith(prefix):
        return None
    match = re.search(r":\s*([\d.]+)\s*(?:tokens/s|s)\s*$", line)
    return float(match.group(1)) if match else None


def parse_gpu_audit_log(text: str) -> GpuAuditLog:
    lines = text.splitlines()
    artifact: str | None = None
    accelerator: str | None = None
    parsed: GpuAuditLog | None = None
    in_unsupported = False

    for line in lines:
        model_match = re.match(r"^Benchmarking model:\s+(\S+)\s+\(", line)
        if model_match:
            artifact = model_match.group(1)
            continue
        backend_match = re.match(r"^Backend\s*:\s*(\w+)\s*$", line)
        if backend_match:
            accelerator = backend_match.group(1)
        if parsed is None:
            if artifact is None or accelerator is None:
                continue
            parsed = GpuAuditLog(artifact=artifact, accelerator=accelerator)

        for attr, prefix in (
            ("prefill_tokens", "Number of tokens in prefill"),
            ("decode_tokens", "Number of tokens in decode"),
            ("max_num_tokens", "Max number of tokens"),
        ):
            value = _header_int(line, prefix)
            if value is not None:
                setattr(parsed, attr, value)
        for attr, prefix in (
            ("prefill_tok_s", "Prefill speed"),
            ("decode_tok_s", "Decode speed"),
            ("init_s", "Init time"),
            ("ttft_s", "Time to first token"),
        ):
            value = _results_float(line, prefix)
            if value is not None:
                setattr(parsed, attr, value)
                parsed.has_results = True

        if line.strip() == _UNSUPPORTED_HEADER:
            in_unsupported = True
            continue
        if in_unsupported:
            partition = _PARTITION_RE.match(line.strip())
            if partition:
                parsed.partition_line = line.strip()
                parsed.delegated_ops = int(partition.group(1))
                remaining = int(partition.group(2))
                parsed.total_ops = parsed.delegated_ops + remaining
                in_unsupported = False
                continue
            op_match = _OP_LINE_RE.match(line)
            if op_match:
                parsed.unsupported_ops.append(op_match.group(1))
                parsed.unsupported_lines.append(_truncate(line))
                continue
        if "Failed to create engine" in line:
            parsed.engine_failed = True
            if parsed.engine_error is None:
                parsed.engine_error = _truncate(line.strip())
        if "Failed to generate content" in line:
            parsed.generate_failures.append(_truncate(line.strip()))

    if parsed is None:
        raise AdapterError(
            "not a gpu_gate_mac.sh log: missing the 'Benchmarking model: …' and "
            "'Backend : …' header lines"
        )
    return parsed


def gpu_audit_record(
    log: GpuAuditLog,
    *,
    env: dict[str, Any],
    date: str,
    speeds_contaminated: bool = False,
) -> dict[str, Any]:
    """One accelerator record from one parsed log. The verdict comes from the
    log body alone — never from an exit code or a summary row."""
    record = _base_record(log.accelerator, env, date)

    if log.engine_failed or not log.has_results:
        record["failure_class"] = "engine_create_failed"
        record["error"] = log.engine_error
        record["evidence"] = [*log.unsupported_lines]
        if log.partition_line is not None:
            record["evidence"].append(log.partition_line)
            record["full_delegation"] = False
            record["delegated_ops"] = log.delegated_ops
            record["total_ops"] = log.total_ops
        if log.engine_error is not None:
            record["evidence"].append(log.engine_error)
        return record

    zero_throughput = (log.decode_tok_s or 0.0) <= 0.0 or (log.prefill_tok_s or 0.0) <= 0.0
    if zero_throughput:
        # The harness prints a results block even when every generate call
        # failed; 0.00 tokens/s is a failure marker, not a measurement.
        record["loads"] = True
        record["failure_class"] = "zero_throughput"
        record["error"] = log.generate_failures[0] if log.generate_failures else None
        record["evidence"] = [
            *log.generate_failures,
            f"results block reported prefill={log.prefill_tok_s:.2f} "
            f"decode={log.decode_tok_s:.2f} tokens/s",
        ]
        if not speeds_contaminated and log.init_s is not None:
            record["metrics"] = {"init_s": log.init_s}
        if speeds_contaminated:
            record["evidence"].append(CONTAMINATED_NOTE)
        return record

    record["loads"] = True
    record["runs"] = True
    metrics: dict[str, float] = {}
    for key, value in (
        ("prefill_tokens", log.prefill_tokens),
        ("decode_tokens", log.decode_tokens),
        ("max_num_tokens", log.max_num_tokens),
    ):
        if value is not None:
            metrics[key] = float(value)
    if speeds_contaminated:
        record["evidence"] = [CONTAMINATED_NOTE]
    else:
        record["prefill_tokens_per_s"] = log.prefill_tok_s
        record["decode_tokens_per_s"] = log.decode_tok_s
        if log.ttft_s is not None and log.ttft_s > 0:
            record["ttft_ms"] = round(log.ttft_s * 1000, 6)
        if log.init_s is not None:
            metrics["init_s"] = log.init_s
        record["evidence"] = [
            f"results block: prefill={log.prefill_tok_s:.2f} decode={log.decode_tok_s:.2f} "
            f"tokens/s, init={log.init_s if log.init_s is not None else '-'} s"
        ]
    if metrics:
        record["metrics"] = metrics
    return record


# --- lane 1b: litert_lm_main / litert-lm benchmark console logs --------------
#
# `gpu_gate_mac.sh` wraps the runtime and prints its own header, which the
# parser above keys on. The Android lane has no wrapper: `litert_lm_main` is
# invoked directly over adb and prints only the runtime's own output, and
# `litert-lm benchmark` on the Mac prints a results block with no header
# either. Those logs therefore state neither the artifact nor the accelerator,
# so BOTH are supplied by the caller instead of being recovered from a file
# name — the same rule the rest of this module follows.
#
# Two rules on what to hand this parser (DECISIONS #163). It is
# last-occurrence-wins on every field, so a file holding several runs is
# recorded as its LAST block under whatever identity the caller names, with no
# error: ingest one slice per (accelerator x condition), never a whole file.
# And only `litert-lm benchmark` output belongs here — the `litert-lm run`
# sections of a shell-trace log are generation-gate output, and gate figures are
# not bench figures.

_CONSOLE_DELEGATE_RE = re.compile(
    r"^VERBOSE: Replacing (\d+) out of (\d+) node\(s\) with delegate "
    r"\((\w+)\) node, yielding (\d+) partitions for subgraph (\d+) \(([^)]*)\)\.$"
)
_CONSOLE_TTFT_RE = re.compile(r"^\s*Time to first token:\s+([\d.]+) s$")
_CONSOLE_SPEED_RE = re.compile(r"^\s*(Prefill|Decode) Speed:\s+([\d.]+) tokens/sec\.$")
_CONSOLE_PROCESSED_RE = re.compile(
    r"^\s*(Prefill|Decode) Turn \d+: Processed (\d+) tokens in "
)
_CONSOLE_INIT_RE = re.compile(r"^\s*- Init Total:\s+([\d.]+) ms$")
_CONSOLE_ABORT_MARKERS = (
    "Check failed: MainHelper",
    "Failed to create engine",
    "engine_advanced_impl.cc",
)


@dataclass
class ConsoleLog:
    """Parsed `litert_lm_main` / `litert-lm benchmark` console output.

    Delegation counts come from the runtime's own `Replacing N out of M` lines,
    counted per accelerator delegate: a run reports one line per subgraph, and
    a partially delegated graph reports a second set from the CPU delegate that
    picks up the remainder. Only lines from the delegate the caller names are
    counted, so a CPU fallback can never be read as GPU residency (the mistake
    that put wrong residency figures on 17 zoo cards on 2026-08-11).
    """

    accelerator: str
    delegate_tag: str | None = None
    prefill_tokens: int | None = None
    decode_tokens: int | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    ttft_s: float | None = None
    init_ms: float | None = None
    has_results: bool = False
    engine_failed: bool = False
    engine_error: str | None = None
    unsupported_ops: list[str] = field(default_factory=list)
    unsupported_lines: list[str] = field(default_factory=list)
    partition_line: str | None = None
    delegated_ops: int | None = None
    total_ops: int | None = None
    subgraph_lines: list[str] = field(default_factory=list)


def parse_console_log(text: str, *, delegate_tag: str = "LITERT_CL") -> ConsoleLog:
    """Parse a raw runtime console log. `delegate_tag` names the delegate whose
    `Replacing …` lines count as delegated (e.g. LITERT_CL, TfLiteXNNPackDelegate)."""
    log = ConsoleLog(accelerator="", delegate_tag=delegate_tag)
    in_unsupported = False
    seen_any_line = False

    for line in text.splitlines():
        seen_any_line = True
        stripped = line.strip()

        speed = _CONSOLE_SPEED_RE.match(line)
        if speed:
            value = float(speed.group(2))
            if speed.group(1) == "Prefill":
                log.prefill_tok_s = value
            else:
                log.decode_tok_s = value
            log.has_results = True
            continue
        processed = _CONSOLE_PROCESSED_RE.match(line)
        if processed:
            count = int(processed.group(2))
            if processed.group(1) == "Prefill":
                log.prefill_tokens = count
            else:
                log.decode_tokens = count
            continue
        ttft = _CONSOLE_TTFT_RE.match(line)
        if ttft:
            log.ttft_s = float(ttft.group(1))
            continue
        init = _CONSOLE_INIT_RE.match(line)
        if init:
            log.init_ms = float(init.group(1))
            continue

        # `litert-lm benchmark` prints the same numbers with different casing
        # and no per-turn lines; reuse the gpu_audit line readers for those.
        for attr, prefix in (
            ("prefill_tokens", "Number of tokens in prefill"),
            ("decode_tokens", "Number of tokens in decode"),
        ):
            value_i = _header_int(line, prefix)
            if value_i is not None:
                setattr(log, attr, value_i)
        for attr, prefix in (
            ("prefill_tok_s", "Prefill speed"),
            ("decode_tok_s", "Decode speed"),
        ):
            value_f = _results_float(line, prefix)
            if value_f is not None:
                setattr(log, attr, value_f)
                log.has_results = True

        delegate = _CONSOLE_DELEGATE_RE.match(stripped)
        if delegate:
            if delegate.group(3) == delegate_tag:
                taken, total = int(delegate.group(1)), int(delegate.group(2))
                log.subgraph_lines.append(stripped)
                # Report the largest subgraph seen: prefill and decode graphs
                # are separate signatures of one model, and summing them would
                # invent a node count that no single graph has.
                if log.total_ops is None or total > log.total_ops:
                    log.delegated_ops, log.total_ops = taken, total
            continue

        if stripped == _UNSUPPORTED_HEADER:
            in_unsupported = True
            continue
        if in_unsupported:
            partition = _PARTITION_RE.match(stripped)
            if partition:
                log.partition_line = stripped
                in_unsupported = False
                continue
            op_match = _OP_LINE_RE.match(line)
            if op_match:
                log.unsupported_ops.append(op_match.group(1))
                log.unsupported_lines.append(_truncate(line))
                continue
            in_unsupported = False

        if any(marker in line for marker in _CONSOLE_ABORT_MARKERS):
            log.engine_failed = True
            if log.engine_error is None:
                log.engine_error = _truncate(stripped)

    if not seen_any_line:
        raise AdapterError("empty log: nothing to ingest")
    if not (log.has_results or log.engine_failed or log.unsupported_ops or log.subgraph_lines):
        raise AdapterError(
            "not a litert_lm_main / litert-lm benchmark console log: found no "
            "speed block, no 'Replacing N out of M' delegate line, and no "
            "engine failure"
        )
    return log


def console_record(
    log: ConsoleLog,
    *,
    accelerator: str,
    env: dict[str, Any],
    date: str,
    prefill_tokens: int | None = None,
    decode_tokens: int | None = None,
) -> dict[str, Any]:
    """One accelerator record from one console log. The verdict comes from the
    log body — a run that produced a speed block ran, one that aborted did not.

    `prefill_tokens` / `decode_tokens` carry the prompt-length condition for
    logs whose header lines never reached the file — a `tail -N` capture that
    kept the results block and cut the `Number of tokens in …` lines above it.
    They are caller-supplied in the same sense as `accelerator` and the
    artifact: the caller states what the recorded evidence says, and only that
    (the benchmark command line traced in the same log, say). They fill in only
    what the log does not state; a value contradicting a count the log itself
    carries is refused, because the log outranks the command line."""
    supplied: list[str] = []
    for name, parsed, given in (
        ("prefill", log.prefill_tokens, prefill_tokens),
        ("decode", log.decode_tokens, decode_tokens),
    ):
        if given is None:
            continue
        if parsed is not None and parsed != given:
            raise AdapterError(
                f"--{name}-tokens {given} contradicts the log's own "
                f"'Number of tokens in {name}' header ({parsed}): the log states "
                "the condition itself, so either the supplied count or the slice is wrong"
            )
        if parsed is None:
            supplied.append(f"{name}={given}")

    record = _base_record(accelerator, env, date)

    if log.delegated_ops is not None and log.total_ops is not None:
        record["delegated_ops"] = log.delegated_ops
        record["total_ops"] = log.total_ops
        record["full_delegation"] = log.delegated_ops == log.total_ops
        record["evidence"].extend(log.subgraph_lines[:4])

    if log.unsupported_lines:
        record["evidence"].extend(log.unsupported_lines)
    if log.partition_line is not None:
        record["evidence"].append(log.partition_line)

    if log.engine_failed or not log.has_results:
        record["failure_class"] = "engine_create_failed"
        record["error"] = log.engine_error
        if log.engine_error is not None:
            record["evidence"].append(log.engine_error)
        return record

    record["loads"] = True
    record["runs"] = True
    record["prefill_tokens_per_s"] = log.prefill_tok_s
    record["decode_tokens_per_s"] = log.decode_tok_s
    if log.ttft_s is not None and log.ttft_s > 0:
        record["ttft_ms"] = round(log.ttft_s * 1000, 6)
    metrics: dict[str, float] = {}
    for key, parsed, given in (
        ("prefill_tokens", log.prefill_tokens, prefill_tokens),
        ("decode_tokens", log.decode_tokens, decode_tokens),
    ):
        value = parsed if parsed is not None else given
        if value is not None:
            metrics[key] = float(value)
    if log.init_ms is not None:
        metrics["init_s"] = round(log.init_ms / 1000.0, 6)
    if metrics:
        record["metrics"] = metrics
    record["evidence"].append(
        f"results block: prefill={log.prefill_tok_s} decode={log.decode_tok_s} tokens/s"
    )
    if supplied:
        record["evidence"].append(
            "token counts supplied by the caller from recorded evidence "
            f"(this log states none): {' '.join(supplied)}"
        )
    return record


# --- lane 2: devicemark ------------------------------------------------------

#: Harness identity from the devicemark dataset card — documented there, not
#: in the row, so the adapter takes it from the card rather than inventing it.
_DEVICEMARK_HARNESS_NOTE = (
    "devicemark leaderboard row (PipelinedBench: warm-state S=1 pipelined decode, "
    "128-token prompt / 256-token decode, two trials, settled device, "
    "numerics-gated; per the devicemark dataset card)"
)


def device_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
    if not slug or not re.match(r"^[a-z0-9]", slug):
        raise AdapterError(f"cannot derive a device slug from {name!r}")
    return slug


@dataclass(frozen=True)
class DevicemarkResult:
    model_id: str
    device: str
    quantization: str
    artifact: str
    record: dict[str, Any]


def devicemark_records(
    text: str,
    *,
    runtime_version: str,
    date: str,
    accelerator: str,
    machine_label: str | None = None,
) -> tuple[list[DevicemarkResult], list[str]]:
    """(ingestible results, skip notes). Non-litertlm rows are skipped — they
    belong in a card's cross_runtime lane — and malformed rows are errors."""
    results: list[DevicemarkResult] = []
    skipped: list[str] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"line {i + 1}: not valid JSON: {exc}") from exc
        artifact_id = row.get("artifact_id", "")
        parts = artifact_id.split("__")
        if len(parts) != 3:
            raise AdapterError(
                f"line {i + 1}: artifact_id {artifact_id!r} is not <slug>__<quant>__<format> "
                "(the devicemark dataset-card contract)"
            )
        slug, quant, fmt = parts
        if fmt != "litertlm" or row.get("runtime") != "litertlm":
            skipped.append(
                f"{artifact_id} on {row.get('device')}: runtime {row.get('runtime')!r} — "
                "not the LiteRT-LM lane (cross-runtime rows belong in a card's "
                "cross_runtime list)"
            )
            continue
        decode = row.get("decode_tok_s")
        if not isinstance(decode, int | float) or decode <= 0:
            raise AdapterError(
                f"line {i + 1}: decode_tok_s {decode!r} — no staged sample shows a "
                "non-positive leaderboard row; refusing to guess what it means"
            )
        env = make_env(
            device=str(row["device"]),
            runtime="litert-lm",
            runtime_version=runtime_version,
            machine_label=machine_label,
        )
        record = _base_record(accelerator, env, date)
        record["loads"] = True
        record["runs"] = True
        record["decode_tokens_per_s"] = float(decode)
        metrics: dict[str, float] = {}
        peak = row.get("peak_mem_mb")
        if isinstance(peak, int | float):
            if row.get("mem_measured") is True:
                record["peak_mem_mb"] = float(peak)
            else:
                # Estimated memory never lands in the measured field.
                metrics["peak_mem_est_mb"] = float(peak)
        power = row.get("power_w")
        if isinstance(power, int | float):
            metrics["power_w"] = float(power)
        if metrics:
            record["metrics"] = metrics
        record["evidence"] = [f"{_DEVICEMARK_HARNESS_NOTE}: {canonical_dumps(row).strip()}"]
        results.append(
            DevicemarkResult(
                model_id=slug,
                device=device_slug(str(row["device"])),
                quantization=quant,
                artifact=artifact_id,
                record=record,
            )
        )
    return results, skipped


# --- lane 3: compat_check ----------------------------------------------------


@dataclass(frozen=True)
class CompatCheckResult:
    model_id: str
    artifact: str
    record: dict[str, Any]
    repo: str = ""  # HF repo tail (e.g. LFM2.5-1.2B-Thinking); "" on older constructors


def compat_check_records(
    doc: dict[str, Any],
    *,
    date: str,
    device: str,
    accelerator: str,
    soc: str | None = None,
    vendor_sdk: str | None = None,
    os_build: str | None = None,
    machine_label: str | None = None,
    model_id_for: dict[str, str] | None = None,
) -> tuple[str, list[CompatCheckResult], list[str]]:
    """-> (runtime_version, ingestible results, refusal notes).

    `model_id_for` maps "<repo tail>/<file>" or "<file>" to the id a result is filed
    under, overriding the mechanical id. Added 2026-09-08 (DECISIONS #168): the
    per-artifact suffix rule below is applied per REPORT, so a repo that ships int4 +
    int8 gets suffixed ids in one report and the bare id in a report that carried
    only one of them — continuity across snapshots needs the caller to say so.

    Refuse-until-sampled (owner decision #3): only `status: "ok"` on the
    `llm` lane has a staged real sample. Every other status or lane is
    refused by name — the owner supplies a real sample when one occurs,
    which unblocks that branch against real strings instead of invented ones.
    """
    runtime_version = doc.get("runtime_version")
    if not isinstance(runtime_version, str) or not runtime_version:
        raise AdapterError("compat_check report carries no runtime_version")
    runtime_note = str(doc.get("runtime", ""))
    results: list[CompatCheckResult] = []
    refused: list[str] = []
    for result in doc.get("results", []):
        repo = str(result.get("repo", ""))
        file = str(result.get("file", ""))
        label = f"{repo}/{file}"
        lane = result.get("lane")
        status = result.get("status")
        if lane != "llm":
            refused.append(
                f"{label}: lane {lane!r} has no staged sample — refused "
                "(supply one real sample to unblock this branch; owner decision #3)"
            )
            continue
        if status != "ok":
            refused.append(
                f"{label}: status {status!r} has no staged sample — refused "
                "(supply one real BROKEN/SUSPECT sample to unblock this branch; "
                "owner decision #3)"
            )
            continue
        # Mechanical lowercasing only (HF repo names are typically cased:
        # LFM2.5-1.2B-Instruct -> lfm2.5-1.2b-instruct); no other renaming.
        # Legacy hand-chosen ids from the console-log lane (lfm25-12b-*) are
        # NOT aliased here — joins across lanes are a rendering concern.
        model_id = repo.split("/")[-1].lower()
        if not re.match(r"^[a-z0-9][a-z0-9._-]*$", model_id):
            refused.append(f"{label}: cannot derive a model id from repo {repo!r}")
            continue
        env = make_env(
            device=device,
            runtime="litert-lm",
            runtime_version=runtime_version,
            soc=soc,
            vendor_sdk=vendor_sdk,
            os_build=os_build,
            machine_label=machine_label,
        )
        record = _base_record(accelerator, env, date)
        record["loads"] = True
        record["runs"] = True
        answer = result.get("answer")
        record["evidence"] = [
            f"compat_check status ok (runtime {runtime_note} {runtime_version}); "
            f"fixed-question answer: {answer!r}"
        ]
        results.append(
            CompatCheckResult(model_id=model_id, artifact=file, record=record, repo=repo.split("/")[-1])
        )
    filed = _disambiguate_artifacts(results, refused)
    if model_id_for:
        filed = [
            replace(res, model_id=override) if (
                override := model_id_for.get(f"{res.repo}/{res.artifact}") or model_id_for.get(res.artifact)
            ) else res
            for res in filed
        ]
    return runtime_version, filed, refused


def _disambiguate_artifacts(
    results: list[CompatCheckResult], refused: list[str]
) -> list[CompatCheckResult]:
    """Give per-artifact ids to repos that carry several ok artifacts.

    Real sample: compat_0.16.0.json ships granite-4.0-h-350m fp16 AND int8 —
    one record file holds one artifact, so the bare repo id collides. The
    suffix is the artifact stem's remainder after the repo id (fp16 / int8),
    matching the console-log lane's quant-suffixed ids (lfm25-vl-450m-int4 /
    -int8). Single-artifact repos keep the bare id for continuity with
    earlier snapshots. Suffixes that cannot be derived, or still collide,
    are refused — never invented.
    """
    counts: dict[str, int] = {}
    for res in results:
        counts[res.model_id] = counts.get(res.model_id, 0) + 1
    out: list[CompatCheckResult] = []
    seen: set[str] = set()
    for res in results:
        model_id = res.model_id
        if counts[res.model_id] > 1:
            stem = res.artifact.rsplit(".", 1)[0].lower()
            suffix = stem[len(model_id):] if stem.startswith(model_id) else stem
            suffix = re.sub(r"[^a-z0-9._-]", "-", suffix.replace("_", "-")).strip("-._")
            if not suffix:
                refused.append(
                    f"{res.model_id}/{res.artifact}: repo carries several ok artifacts "
                    "but no id suffix can be derived from this artifact name — refused"
                )
                continue
            model_id = f"{res.model_id}-{suffix}"
        if model_id in seen:
            refused.append(
                f"{res.model_id}/{res.artifact}: derived id {model_id!r} collides with "
                "another artifact in this report — refused"
            )
            continue
        seen.add(model_id)
        out.append(replace(res, model_id=model_id))
    return out


# --- lane 5: iOS bench-app "yardstick" result JSON -----------------------------
#
# Mapped against the real samples staged under data/examples/llm_samples/yardstick/
# (owner's iOS BenchmarkApp, `--yardstick-autorun`, 2026-08-19). What the JSON
# carries and what it does not:
#   carries : timestamp (UTC ISO), task id, outputSample, parameters, runtime name,
#             device.modelIdentifier/systemVersion/physicalMemoryMB,
#             metrics.{promptTokensPerSecond, decodeTokensPerSecond,
#             firstTokenLatencyMS, memoryPeakDuringDecodeMB (device-measured),
#             memoryAfterGenerationMB, loadTimeSeconds, promptTokenCount,
#             generatedTokenCount, stopReason, coldRun}
#   does NOT: the runtime VERSION, the accelerator (the app's `--litert-cpu` flag —
#             absent means the default, which for litert-lm on iOS is the Metal GPU),
#             the published artifact (model.primaryFile is the staged copy's name,
#             model.displayName/quantization are the app's catalog text and can lag
#             the file), and any expected answer for the vision probes.
# So runtime version, accelerator, artifact and the catalog model id are
# caller-supplied; the date comes from the source's own UTC timestamp unless the
# caller overrides it; output_match stays null (the expected answer is not in the
# file — the outputSample is carried verbatim as evidence for a human to read).
# Several task files for one (model, device, accelerator) merge into ONE record:
# throughput/TTFT from the task that generated the most tokens (named in the
# evidence), peak memory = the max device-measured peak across tasks, and every
# task's numbers kept in `metrics` under a task-prefixed key.

_YARDSTICK_REQUIRED = ("timestamp", "task", "runtime", "metrics", "device")


@dataclass(frozen=True)
class YardstickRun:
    task: str
    timestamp: str
    prompt_tokens: int | None
    generated_tokens: int | None
    prefill_tok_s: float | None
    decode_tok_s: float | None
    ttft_ms: float | None
    peak_mem_mb: float | None
    load_s: float | None
    output_sample: str | None
    stop_reason: str | None
    device_model: str | None
    os_version: str | None
    cold: bool | None = None
    context_tokens: int | None = None


def parse_yardstick(doc: dict[str, Any]) -> YardstickRun:
    missing = [k for k in _YARDSTICK_REQUIRED if k not in doc]
    if missing:
        raise AdapterError(f"not a yardstick result: missing {', '.join(missing)}")
    if doc.get("runtime") != "litert-lm":
        raise AdapterError(
            f"runtime {doc.get('runtime')!r} is not the litert-lm lane "
            "(other runtimes belong in a card's cross_runtime list)"
        )
    m = doc["metrics"]
    dev = doc.get("device") or {}

    def num(key: str) -> float | None:
        v = m.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return YardstickRun(
        task=str(doc["task"]),
        timestamp=str(doc["timestamp"]),
        prompt_tokens=int(m["promptTokenCount"]) if "promptTokenCount" in m else None,
        generated_tokens=int(m["generatedTokenCount"]) if "generatedTokenCount" in m else None,
        prefill_tok_s=num("promptTokensPerSecond"),
        decode_tok_s=num("decodeTokensPerSecond"),
        ttft_ms=num("firstTokenLatencyMS"),
        peak_mem_mb=num("memoryPeakDuringDecodeMB"),
        load_s=num("loadTimeSeconds"),
        output_sample=doc.get("outputSample"),
        stop_reason=m.get("stopReason"),
        device_model=dev.get("modelIdentifier"),
        os_version=dev.get("systemVersion"),
        cold=m["coldRun"] if isinstance(m.get("coldRun"), bool) else None,
        context_tokens=int(m["contextTokensConfigured"])
        if isinstance(m.get("contextTokensConfigured"), (int, float))
        and not isinstance(m.get("contextTokensConfigured"), bool)
        else None,
    )


def yardstick_date(runs: list[YardstickRun]) -> str:
    """The source-stamped measurement date (UTC calendar day of the earliest
    timestamp). Refuses a set that spans days — one record, one date."""
    days = sorted({r.timestamp[:10] for r in runs})
    if len(days) != 1:
        raise AdapterError(f"task files span several UTC days {days}; ingest per day")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", days[0]):
        raise AdapterError(f"unparseable timestamp {runs[0].timestamp!r}")
    return days[0]


def yardstick_record(
    runs: list[YardstickRun],
    *,
    accelerator: str,
    env: dict[str, Any],
    date: str,
) -> dict[str, Any]:
    if not runs:
        raise AdapterError("no task results to merge")
    # The bench protocol (cold-warm-split) runs ONE task several times in one
    # process: run 1 cold, runs >= 2 warm, headline = the warm median (the
    # vendor-card convention). Several runs of one task therefore AGGREGATE
    # into that task's figures — every raw run stays in the evidence, the cold
    # figure and the warm spread stay in metrics. (Owner-approved 2026-08-27;
    # before that, duplicate task ids were refused.)
    groups: dict[str, list[YardstickRun]] = {}
    for r in runs:
        groups.setdefault(r.task, []).append(r)

    def _median(values: list[float | None]) -> float | None:
        known = [v for v in values if v is not None]
        return statistics.median(known) if known else None

    merged: list[YardstickRun] = []
    extra_metrics: dict[str, float] = {}
    agg_notes: list[str] = []
    for task, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        group = sorted(group, key=lambda r: r.timestamp)
        warms = [r for r in group if r.cold is False]
        colds = [r for r in group if r.cold is True]
        pool, pool_name = (warms, "warm") if warms else (group, "all")
        rep = replace(
            pool[-1],
            prefill_tok_s=_median([r.prefill_tok_s for r in pool]),
            decode_tok_s=_median([r.decode_tok_s for r in pool]),
            ttft_ms=_median([r.ttft_ms for r in pool]),
            peak_mem_mb=max(
                (r.peak_mem_mb for r in group if r.peak_mem_mb is not None),
                default=None,
            ),
        )
        merged.append(rep)
        key = task.replace("-", "_")
        extra_metrics[f"{key}.n_runs"] = float(len(group))
        decodes = [r.decode_tok_s for r in pool if r.decode_tok_s is not None]
        if len(decodes) > 1 and rep.decode_tok_s:
            extra_metrics[f"{key}.decode_spread_pct"] = round(
                (max(decodes) - min(decodes)) / rep.decode_tok_s * 100, 2
            )
        cold_decode = _median([r.decode_tok_s for r in colds])
        if cold_decode is not None:
            extra_metrics[f"{key}.cold_decode_tokens_per_s"] = round(cold_decode, 6)
        agg_notes.append(
            f"'{task}': {len(group)} runs of one task — throughput = median of "
            f"{len(pool)} {pool_name} run(s)"
            + (
                f"; cold decode kept in metrics ({len(colds)} cold run(s))"
                if colds
                else "; runs carry no cold/warm flag"
                if not warms
                else ""
            )
        )
    record = _base_record(accelerator, env, date)
    record["loads"] = True
    record["runs"] = True
    # Throughput from the longest generation (short vision probes emit 2 tokens and
    # their decode rate is not a throughput figure).
    primary = max(merged, key=lambda r: (r.generated_tokens or 0))
    # A per-cell context budget (e.g. LFM2.5's exported-prefill-plan 1024) is
    # stamped per run; it reaches the record only when every run agrees.
    ctxs = {r.context_tokens for r in groups[primary.task] if r.context_tokens is not None}
    if len(ctxs) == 1:
        record["context_length"] = ctxs.pop()
    # tok/s rounded to 2 decimals (the app prints 15 digits of float noise);
    # the unrounded values stay in metrics.
    if primary.prefill_tok_s is not None:
        record["prefill_tokens_per_s"] = round(primary.prefill_tok_s, 2)
    if primary.decode_tok_s is not None:
        record["decode_tokens_per_s"] = round(primary.decode_tok_s, 2)
    if primary.ttft_ms is not None:
        record["ttft_ms"] = primary.ttft_ms
    peaks = [r.peak_mem_mb for r in runs if r.peak_mem_mb is not None]
    if peaks:
        record["peak_mem_mb"] = round(max(peaks), 3)
    metrics: dict[str, float] = {}
    for r in merged:
        k = r.task.replace("-", "_")
        for name, value in (
            ("prompt_tokens", r.prompt_tokens),
            ("generated_tokens", r.generated_tokens),
            ("prefill_tokens_per_s", r.prefill_tok_s),
            ("decode_tokens_per_s", r.decode_tok_s),
            ("ttft_ms", r.ttft_ms),
            ("peak_mem_mb", r.peak_mem_mb),
            ("load_s", r.load_s),
        ):
            if value is not None:
                metrics[f"{k}.{name}"] = round(float(value), 6)
    metrics.update(extra_metrics)
    record["metrics"] = metrics
    record["evidence"].append(
        f"throughput taken from task '{primary.task}' ({primary.generated_tokens} generated "
        f"tokens, {primary.timestamp}); peak_mem_mb = max memoryPeakDuringDecodeMB over "
        f"{len(runs)} task file(s)"
    )
    record["evidence"].extend(agg_notes)
    for r in runs:
        sample = (r.output_sample or "").replace("\n", "\\n")
        record["evidence"].append(_truncate(
            f"[{r.task} {r.timestamp}] prefill={r.prefill_tok_s} decode={r.decode_tok_s} "
            f"ttft_ms={r.ttft_ms} peak_mb={r.peak_mem_mb} stop={r.stop_reason} "
            f"output={sample!r}"
        ))
    return record


# --- lane 6: npubench classic sweep (S26 NPU/GPU, JIT + AOT) -----------------


#: The measurement-acceptance contract the sweep driver enforced; carried on
#: every measured record so a reader knows what "measured" meant here.
_NPUBENCH_CONDITIONS_NOTE = (
    "npubench sweep row: N=50 median, 1 backend = 1 process, accepted only with "
    "thermal NONE->NONE; NPU rows additionally required qnn_partition delegate "
    "evidence in logcat (a stock model asked for on the NPU can silently land on "
    "XNNPACK and report a CPU number)"
)

#: error-string -> failure_class, mapped against the staged real sample only.
#: Order matters: first prefix match wins.
_NPUBENCH_FAILURES = [
    ("AOT compile failed on host", "aot_compile_failed", False, False),
    ("no delegate evidence in logcat", "silent_cpu_fallback", True, True),
    ("host_timeout", "timeout", False, False),
    ("process_crashed", "process_crashed", False, False),
    ("LiteRtException: Failed to compile model", "compile_failed", False, False),
    ("LiteRtException: Failed to load model from file", "load_failed", False, False),
    ("LiteRtException: Failed to invoke the compiled model", "invoke_failed", True, False),
]


@dataclass(frozen=True)
class NpubenchResult:
    model_id: str
    artifact: str
    date: str
    records: list[dict[str, Any]]


def _npubench_slug(repo: str, file: str, multi_file: bool) -> str:
    """The cards-lane id convention: repo name lowercased with `-LiteRT`
    dropped; repos shipping several .tflite files append `__<stem>` so one
    record file names one artifact (the devicemark collision decision)."""
    base = re.sub(r"-litert$", "", repo.split("/", 1)[-1].lower())
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if not multi_file:
        return base
    stem = file.rsplit("/", 1)[-1]
    stem = re.sub(r"\.tflite$", "", stem).lower()
    return f"{base}__{re.sub(r'[^a-z0-9._-]+', '-', stem)}"


#: `evidence.replacing` delegate tag per journal backend — the residency line
#: the driver captured from logcat, when it captured one.
_NPUBENCH_REPLACING_TAG = {"gpu": "LITERT_CL", "cpu": "TfLiteXNNPackDelegate"}


def _replacing_counts(row: dict[str, Any], backend: str) -> tuple[int, int] | None:
    """(delegated, total) from the journal's captured `Replacing N out of M
    node(s) with delegate (<tag>)` lines for this backend's delegate — the
    largest subgraph, the console-log convention. None when the row carries
    no such line (older journals) or the tag does not match."""
    tag = _NPUBENCH_REPLACING_TAG.get(backend)
    entries = (row.get("evidence") or {}).get("replacing") or []
    best: tuple[int, int] | None = None
    for entry in entries:
        if len(entry) != 3 or entry[2] != tag:
            continue
        taken, total = int(entry[0]), int(entry[1])
        if best is None or total > best[1]:
            best = (taken, total)
    return best


def npubench_records(
    text: str,
    *,
    runtime_version: str,
    device: str,
    device_name: str,
    soc: str | None = None,
    os_build: str | None = None,
    machine_label: str | None = None,
    repo_for: dict[str, str] | None = None,
) -> tuple[list[NpubenchResult], list[str]]:
    """Replay a `s2_npu_sweep/results.jsonl` journal into device-run records.

    The journal is append-only and self-correcting, and the replay is faithful
    to the driver's own semantics (mapped against the staged real sample):

    - `annul` rows void everything earlier for that backend (`all` = both
      backends and any skip/unobtainable state) — contention windows and
      false verdicts stay in the journal but never in a record.
    - measured records come ONLY from accepted rows (thermal NONE->NONE, NPU
      delegate evidence); hot/unaccepted rows are excluded entirely.
    - NPU: the JIT `cached` phase (or the `aot` phase) is the measurement; an
      accepted `cold` row contributes `metrics.jit_first_load_ms` (its load
      includes the on-device compile). `mode` is carried in metrics and the
      AOT/JIT split also lands in `env.vendor_sdk` — the two modes must never
      share a table unlabeled.
    - failures map to failure_class by exact prefix (see _NPUBENCH_FAILURES);
      an error string matching no staged shape is a refusal, not a guess.
    - `skipped` (soc-mismatch) and `unobtainable` rows produce no record.
    - a journal whose rows carry no `repo`/`slug` (the Pixel 8a `run_p8a.py`
      shape, staged 2026-09-02) is ingested only with `repo_for`, a
      file -> repo map the caller takes from the sweep's own README; a row
      that does name a repo must agree with the map.
    - `cpu` backend rows (CompiledModel CPU + XNNPACK, the same journal) land
      as `cpu_xnnpack`; delegation counts come from the captured
      `evidence.replacing` line when the row carries one.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    repo_files: dict[str, set[str]] = {}
    skipped: list[str] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"line {i + 1}: not valid JSON: {exc}") from exc
        repo, slug, file = row.get("repo"), row.get("slug"), row.get("file")
        mapped = (repo_for or {}).get(file) if file else None
        if repo and mapped and repo != mapped:
            raise AdapterError(
                f"line {i + 1}: row names repo {repo!r} but --repo-for maps {file!r} to {mapped!r}"
            )
        repo = repo or mapped
        if not slug and repo and file:
            slug = f"{repo.split('/', 1)[-1]}__{file.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
        if not repo or not slug:
            raise AdapterError(
                f"line {i + 1}: row carries no repo/slug"
                + (f" (no --repo-for mapping for {file!r})" if file else "")
            )
        if file:
            repo_files.setdefault(repo, set()).add(file)
        e = by_key.setdefault((repo, slug), {"file": file})
        if row.get("annul"):
            backends = ("npu", "gpu", "cpu") if row["backend"] == "all" else (row["backend"],)
            for b in backends:
                for k in (b, b + "_fail", b + "_cold"):
                    e.pop(k, None)
            if row["backend"] == "all":
                e.pop("skipped", None)
                e.pop("unobtainable", None)
            continue
        if row.get("skipped"):
            e["skipped"] = row["skipped"]
            continue
        if row.get("verdict") == "unobtainable":
            e["unobtainable"] = row.get("error") or "unobtainable"
            continue
        b, ph = row.get("backend"), row.get("phase")
        if row.get("ok") and row.get("accepted"):
            if b == "npu" and ph == "cold":
                e["npu_cold"] = row
            elif b == "npu" and ph in ("cached", "aot"):
                e["npu"] = row
                e.pop("npu_fail", None)
            elif b in ("gpu", "cpu"):
                e[b] = row
                e.pop(b + "_fail", None)
        elif row.get("ok") is False and row.get("error"):
            # a cached-phase host timeout is the driver's pacing guess being
            # wrong, not the model failing (the cold run already ran)
            if ph == "cached" and str(row["error"]).startswith("host_timeout"):
                continue
            if b in ("npu", "gpu", "cpu") and b not in e:
                e.setdefault(b + "_fail", row)

    results: list[NpubenchResult] = []
    for (repo, slug), e in sorted(by_key.items()):
        if e.get("skipped") or e.get("unobtainable"):
            skipped.append(f"{slug}: {e.get('skipped') or e.get('unobtainable')}")
            continue
        multi = len(repo_files.get(repo, set())) > 1
        model_id = _npubench_slug(repo, e.get("file") or slug, multi)
        records: list[dict[str, Any]] = []
        dates: list[str] = []
        for backend, accel in (
            ("npu", "npu_qnn"), ("gpu", "gpu_mldrift"), ("cpu", "cpu_xnnpack")
        ):
            row = e.get(backend)
            fail = e.get(backend + "_fail")
            src = row or fail
            if src is None:
                continue
            mode = src.get("mode") or ("aot" if src.get("phase") == "aot" else "jit")
            vendor = None
            if backend == "npu":
                vendor = (
                    "QAIRT (Hexagon, AOT host-compiled, SM8850 target)"
                    if mode == "aot"
                    else "QAIRT (Hexagon, JIT on-device)"
                )
            env = make_env(
                device=device_name,
                runtime="litert",
                runtime_version=runtime_version,
                soc=soc,
                vendor_sdk=vendor,
                os_build=os_build,
                machine_label=machine_label,
            )
            date = str(src.get("ts", ""))[:10]
            if not date:
                raise AdapterError(f"{slug}/{backend}: row carries no ts date")
            record = _base_record(accel, env, date)
            if row is not None:
                record["loads"] = True
                record["runs"] = True
                record["latency_p50_ms"] = float(row["median_ms"])
                metrics: dict[str, Any] = {
                    "iterations": int(row.get("runs") or row.get("iters") or 0),
                    "latency_min_ms": float(row["min_ms"]),
                    "latency_max_ms": float(row["max_ms"]),
                    "load_ms": float(row["load_ms"]),
                }
                cold = e.get("npu_cold") if backend == "npu" else None
                if cold is not None and mode == "jit":
                    metrics["jit_first_load_ms"] = float(cold["load_ms"])
                qnn = (row.get("evidence") or {}).get("qnn_partition")
                if backend == "npu" and qnn is not None:
                    metrics["qnn_partition_lines"] = int(qnn)
                record["metrics"] = metrics
                record["evidence"] = [
                    _NPUBENCH_CONDITIONS_NOTE,
                    f"mode={mode} thermal={row.get('thermal')} "
                    f"headroom={row.get('headroom')} attempt={row.get('attempt')}",
                ]
                counts = _replacing_counts(row, backend)
                if counts is not None:
                    record["delegated_ops"], record["total_ops"] = counts
                    record["full_delegation"] = counts[0] == counts[1]
                    record["evidence"].append(
                        f"logcat: Replacing {counts[0]} out of {counts[1]} node(s) with delegate "
                        f"({_NPUBENCH_REPLACING_TAG[backend]}) (largest subgraph captured by the driver)"
                    )
            else:
                error = str(fail["error"])
                for prefix, cls, loads, runs in _NPUBENCH_FAILURES:
                    if error.startswith(prefix) or (
                        "LiteRtException" in prefix and prefix.split(": ", 1)[-1] in error
                    ):
                        record["failure_class"] = cls
                        record["loads"] = loads
                        record["runs"] = runs
                        break
                else:
                    skipped.append(
                        f"{slug}/{backend}: error shape not in the staged sample, "
                        f"refusing to classify: {error[:120]!r}"
                    )
                    continue
                record["error"] = _truncate(error)
                record["evidence"] = [_truncate(error)]
            records.append(record)
            dates.append(date)
        if records:
            results.append(
                NpubenchResult(
                    model_id=model_id,
                    artifact=(e.get("file") or slug).rsplit("/", 1)[-1],
                    date=max(dates),
                    records=records,
                )
            )
    return results, skipped


# --- lane 7: Raspberry Pi 5 classic sweep (benchmark_model, CPU/XNNPACK) ------


_PI5_CONDITIONS_NOTE = (
    "pi5 sweep row: LiteRT benchmark_model, CPU/XNNPACK at --num_threads=4, "
    "3 invocations per file of 10 warm-up + 50 timed runs (the tool caps a phase "
    "at 150 s, so very slow graphs run fewer); latency = median of the three "
    "per-invocation medians over the timed phase; a row counts as measured only "
    "when every invocation exited 0 with XNNPACK engaged and vcgencmd get_throttled "
    "0x0 before and after"
)


def pi5_benchmark_records(
    text: str,
    *,
    device: str,
    device_name: str | None = None,
    soc: str | None = None,
    os_build: str | None = None,
    machine_label: str | None = None,
) -> tuple[list[NpubenchResult], list[str]]:
    """Replay a `pi5_bench.py` results.jsonl (Raspberry Pi 5 card campaign,
    wave 1) into device-run records — mapped against the staged real sample.

    - one row = one (repo, file[, signature]); the runtime version is READ
      from the row's `versions` (ai-edge-litert-nightly, the LiteRT the
      benchmark_model binary links; litert-cli-nightly + the binary's sha256
      go to evidence), never supplied by the caller.
    - a multi-signature file gives one record per signature, carried in the
      record's `signature` field (part of record_label), same model_id.
    - accepted iff every invocation exit == 0, xnnpack true and throttled_after
      == "0x0" (the card driver's own `row_ok`); anything else is skipped with
      a note — error shapes are refused-until-sampled.
    - `footprint_peak_mb` is the tool's measured peak footprint
      (--report_peak_memory_footprint) -> peak_mem_mb.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    repo_files: dict[str, set[str]] = {}
    skipped: list[str] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"line {i + 1}: not valid JSON: {exc}") from exc
        repo, file = row.get("repo"), row.get("file")
        if not repo or not file:
            raise AdapterError(f"line {i + 1}: row carries no repo/file")
        label = f"{repo}/{file}" + (f"#{row['signature']}" if row.get("signature") else "")
        if row.get("error") or row.get("signature_list_error"):
            skipped.append(
                f"{label}: row carries an error ({str(row.get('error') or row.get('signature_list_error'))[:100]!r}) "
                "— no error shape is staged, refusing to classify"
            )
            continue
        invocations = row.get("invocations") or []
        if len(invocations) != 3 or row.get("median_of_medians_us") is None:
            skipped.append(f"{label}: incomplete row ({len(invocations)} invocation(s))")
            continue
        bad = [
            k for k, inv in enumerate(invocations)
            if inv.get("error") or inv.get("exit") != 0 or not inv.get("xnnpack")
            or inv.get("throttled_after") != "0x0"
        ]
        if bad:
            skipped.append(
                f"{label}: invocation(s) {bad} not accepted (exit/xnnpack/throttle) — no record"
            )
            continue
        versions = row.get("versions") or {}
        runtime_version = versions.get("ai-edge-litert-nightly") or versions.get("ai-edge-litert")
        if not runtime_version:
            raise AdapterError(f"line {i + 1}: row's versions carry no ai-edge-litert version")
        date = str(row.get("ts", ""))[:10]
        if not date:
            raise AdapterError(f"line {i + 1}: row carries no ts")
        repo_files.setdefault(repo, set()).add(file)
        e = by_key.setdefault((repo, file), {"records": [], "dates": [], "versions": set()})
        env = make_env(
            device=device_name or versions.get("cpu") or device,
            runtime="litert",
            runtime_version=runtime_version,
            soc=soc,
            os_build=os_build or versions.get("platform"),
            machine_label=machine_label,
        )
        record = _base_record("cpu_xnnpack", env, date)
        record["loads"] = True
        record["runs"] = True
        record["latency_p50_ms"] = round(float(row["median_of_medians_us"]) / 1000.0, 3)
        peaks = [float(inv.get("footprint_peak_mb") or 0) for inv in invocations]
        if max(peaks) > 0:
            record["peak_mem_mb"] = max(peaks)
        metrics: dict[str, Any] = {
            "latency_min_ms": round(float(row["spread_min_us"]) / 1000.0, 3),
            "latency_max_ms": round(float(row["spread_max_us"]) / 1000.0, 3),
            "iterations": int(sum(int(inv.get("runs") or 0) for inv in invocations)),
            "invocations": len(invocations),
            "threads": int(row.get("threads") or 0),
            "warmup_runs_per_invocation": int(row.get("warmup_runs") or 0),
            "init_ms": round(statistics.median(float(inv["init_ms"]) for inv in invocations), 3),
        }
        record["metrics"] = metrics
        if row.get("signature"):
            record["signature"] = str(row["signature"])
        record["evidence"] = [
            _PI5_CONDITIONS_NOTE,
            "versions: " + ", ".join(f"{k}={v}" for k, v in sorted(versions.items())),
            *(
                f"invocation {k}: exit={inv.get('exit')} wall_s={inv.get('wall_s')} "
                f"temp {inv.get('temp_before')}->{inv.get('temp_after')}C "
                f"throttled={inv.get('throttled_after')} xnnpack={inv.get('xnnpack')} "
                f"median_us={inv.get('median_us')} runs={inv.get('runs')} "
                f"footprint_peak_mb={inv.get('footprint_peak_mb')}"
                for k, inv in enumerate(invocations)
            ),
        ]
        if row.get("signature"):
            record["evidence"].append(
                f"--signature_to_run_for={row['signature']} (file signatures: "
                f"{', '.join(row.get('signatures') or [])})"
            )
        e["records"].append(record)
        e["dates"].append(date)
        e["versions"].add(runtime_version)

    results: list[NpubenchResult] = []
    for (repo, file), e in sorted(by_key.items()):
        if len(e["versions"]) != 1:
            skipped.append(f"{repo}/{file}: rows span runtime versions {sorted(e['versions'])} — split the journal")
            continue
        multi = len(repo_files.get(repo, set())) > 1
        results.append(
            NpubenchResult(
                model_id=_npubench_slug(repo, file, multi),
                artifact=file.rsplit("/", 1)[-1],
                date=max(e["dates"]),
                records=e["records"],
            )
        )
    return results, skipped


# --- lane 8: npubench parity gate (keyed JSON report, on-device parity) --------


_PARITY_KEY_RE = re.compile(r"^(?P<variant>.+?)_(?P<dur>\d+s)_(?P<accel>cpu|gpu)$")
_PARITY_CONDITIONS_NOTE = (
    "npubench parity gate row: N=20 median in one process per accelerator, "
    "real audio input, output tensors dumped and diffed against the Mac tflite "
    "run and the fp32 eager transcript; verdict per leg = transcript == eager "
    "AND thermal NONE->NONE"
)


def npubench_parity_records(
    doc: dict[str, Any],
    *,
    repo: str,
    files: dict[str, str],
    runtime_version: str,
    date: str,
    device: str,
    device_name: str,
    signature_for: dict[str, str] | None = None,
    soc: str | None = None,
    os_build: str | None = None,
    machine_label: str | None = None,
) -> tuple[list[NpubenchResult], list[str]]:
    """Map an `s26_gate.py` parity report (`{"<variant>_<dur>_<accel>": {...}}`,
    the npubench `#parity` test) into device-run records.

    The report names neither the artifact files nor the signature names: the
    caller maps variant -> published file (`files`) and duration -> signature
    (`signature_for`, from the logcat's `#transcribe_10s` asset suffix) and
    supplies the date (the report has no timestamp; the logcat does). Parity
    numbers (ids_match_*, logit_maxdiff_mac) go to metrics; output_match stays
    null because the report applies its own criterion, not the shared
    tolerance policy. GPU failures use the npubench error vocabulary.
    """
    by_file: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for key, entry in doc.items():
        match = _PARITY_KEY_RE.match(key)
        if not match:
            raise AdapterError(f"report key {key!r} is not <variant>_<dur>_<accel>")
        variant, dur, accel = match.group("variant"), match.group("dur"), match.group("accel")
        file = files.get(variant)
        if not file:
            raise AdapterError(f"report key {key!r}: no --file mapping for variant {variant!r}")
        signature = (signature_for or {}).get(dur, dur)
        env = make_env(
            device=device_name, runtime="litert", runtime_version=runtime_version,
            soc=soc, os_build=os_build, machine_label=machine_label,
        )
        record = _base_record("cpu" if accel == "cpu" else "gpu_mldrift", env, date)
        record["signature"] = signature
        status = entry.get("status")
        if status == "RAN":
            record["loads"] = True
            record["runs"] = True
            record["latency_p50_ms"] = float(entry["median_ms"])
            metrics: dict[str, Any] = {
                "latency_min_ms": float(entry["min_ms"]),
                "latency_max_ms": float(entry["max_ms"]),
                "load_ms": float(entry["load_ms"]),
                "iterations": int(entry.get("runs") or 0),
            }
            for k in ("ids_match_mac_tflite", "ids_match_eager", "logit_maxdiff_mac"):
                if entry.get(k) is not None:
                    metrics[k] = float(entry[k])
            record["metrics"] = metrics
            record["evidence"] = [
                _PARITY_CONDITIONS_NOTE,
                f"report key {key}: accel={entry.get('accel')} thermal={entry.get('thermal')} "
                f"headroom={entry.get('headroom')} PASS={entry.get('PASS')} "
                f"transcript_match_eager={entry.get('transcript_match_eager')}",
                f"transcript: {_truncate(str(entry.get('transcript', '')))}",
            ]
            if entry.get("PASS") is not True:
                skipped.append(f"{key}: RAN but PASS={entry.get('PASS')!r} — recorded as runs=true, verdict in evidence")
        elif status == "FAILED":
            error = str(entry.get("error", ""))
            for prefix, cls, loads, runs in _NPUBENCH_FAILURES:
                if error.startswith(prefix) or (
                    "LiteRtException" in prefix and prefix.split(": ", 1)[-1] in error
                ):
                    record["failure_class"] = cls
                    record["loads"] = loads
                    record["runs"] = runs
                    break
            else:
                skipped.append(
                    f"{key}: error shape not in the staged sample, refusing to classify: {error[:120]!r}"
                )
                continue
            record["error"] = _truncate(error)
            record["evidence"] = [f"report key {key}: status=FAILED error={_truncate(error)}"]
        else:
            skipped.append(f"{key}: status {status!r} not staged, refusing")
            continue
        by_file.setdefault(file, {"records": []})["records"].append(record)

    results: list[NpubenchResult] = []
    multi = len(files) > 1
    for file, e in sorted(by_file.items()):
        results.append(
            NpubenchResult(
                model_id=_npubench_slug(repo, file, multi),
                artifact=file.rsplit("/", 1)[-1],
                date=date,
                records=e["records"],
            )
        )
    return results, skipped


# --- lane 9: Raspberry Pi 5 LLM sweep (`litert-lm benchmark`, wave 2) -----------

_PI5_LLM_CONDITIONS_NOTE = (
    "pi5 LLM sweep row: `litert-lm benchmark --backend cpu --cpu-thread-count 4 -p 256 -d 256 "
    "--runs 1 --cache memory` (wave-2 driver pi5_llm_bench.py; --cache memory rather than the "
    "house --cache no, which OOM-kills every >=1.2B file on the 8 GB Pi — equivalence measured "
    "on granite-350m int8, +2-3%), 3 invocations per file with cool-down to <=52 C between them, "
    "vcgencmd measure_temp + get_throttled logged per invocation, peak RSS polled from /proc; "
    "throughput = median of the three invocations (spread in metrics); a row counts as measured "
    "only when the real-generation gate (`litert-lm run`, degenerate-output check) passed and "
    "every invocation exited 0 with get_throttled 0x0"
)


def _pi5_llm_model_id(repo: str, file: str, multi_file: bool) -> str:
    """Mechanical id: repo tail lowercased with `-LiteRT` dropped; a repo shipping several
    bundles in the journal appends the artifact stem's remainder after the repo id (the
    compat_check convention: LFM2.5-1.2B-Instruct_int4 -> lfm2.5-1.2b-instruct-int4)."""
    base = re.sub(r"-litert$", "", repo.split("/", 1)[-1].lower())
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if not multi_file:
        return base
    stem = re.sub(r"\.litertlm$", "", file.rsplit("/", 1)[-1]).lower()
    suffix = stem[len(base):] if stem.startswith(base) else stem
    suffix = re.sub(r"[^a-z0-9._-]", "-", suffix.replace("_", "-")).strip("-._")
    return f"{base}-{suffix}" if suffix else base


def pi5_llm_records(
    text: str,
    *,
    device: str,
    device_name: str | None = None,
    soc: str | None = None,
    os_build: str | None = None,
    machine_label: str | None = None,
    model_id_for: dict[str, str] | None = None,
) -> tuple[list[NpubenchResult], list[str]]:
    """Replay a `pi5_llm_bench.py` llm_results.jsonl (Raspberry Pi 5 card campaign,
    wave 2: LLM .litertlm bundles through `litert-lm benchmark`) into device-run
    records — mapped against the staged real rows (data/examples/llm_samples/pi5/).

    - one row = one (repo, file); the runtime version is READ from the row's
      `versions["litert-lm"]` (the wheel the venv ran), never supplied by the caller;
      `versions.cpu` names the board and `versions.platform` the kernel + glibc.
    - accepted iff the row's own gate passed (`gate.status == "pass"`, exit 0), it holds
      three invocations, and every invocation exited 0 with `throttled_after == "0x0"`;
      anything else is skipped with a note — error shapes are refused-until-sampled.
    - throughput = the row's medians (median_prefill_tps / median_decode_tps /
      median_ttft_s), min/max carried in `metrics`; `peak_rss_mb` (process RSS polled
      from /proc) -> peak_mem_mb. Only `backend == "cpu"` is staged.
    - `model_id_for` maps "<repo tail>/<file>" or "<file>" to a catalog id (the cards'
      hand-chosen ids); otherwise the mechanical id above.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    repo_files: dict[str, set[str]] = {}
    skipped: list[str] = []
    rows: list[tuple[int, dict[str, Any]]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"line {i + 1}: not valid JSON: {exc}") from exc
        repo, file = row.get("repo"), row.get("file")
        if not repo or not file:
            raise AdapterError(f"line {i + 1}: row carries no repo/file")
        rows.append((i, row))
        if not row.get("error"):
            repo_files.setdefault(repo, set()).add(file)

    for i, row in rows:
        repo, file = row["repo"], row["file"]
        label = f"{repo}/{file}"
        if row.get("error"):
            skipped.append(
                f"{label}: row carries an error ({str(row['error'])[:100]!r}) — no error shape "
                "is staged, refusing to classify"
            )
            continue
        if row.get("backend") != "cpu":
            skipped.append(f"{label}: backend {row.get('backend')!r} has no staged sample — refused")
            continue
        gate = row.get("gate") or {}
        invocations = row.get("invocations") or []
        if len(invocations) != 3 or row.get("median_decode_tps") is None:
            skipped.append(f"{label}: incomplete row ({len(invocations)} invocation(s))")
            continue
        if gate.get("status") != "pass" or gate.get("exit") != 0:
            skipped.append(
                f"{label}: generation gate not passed (status={gate.get('status')!r}, "
                f"exit={gate.get('exit')!r}) — no record"
            )
            continue
        bad = [
            k for k, inv in enumerate(invocations)
            if inv.get("error") or inv.get("exit") != 0 or inv.get("throttled_after") != "0x0"
        ]
        if bad:
            skipped.append(f"{label}: invocation(s) {bad} not accepted (exit/throttle) — no record")
            continue
        versions = row.get("versions") or {}
        runtime_version = versions.get("litert-lm")
        if not runtime_version:
            raise AdapterError(f"line {i + 1}: row's versions carry no litert-lm version")
        date = str(row.get("ts", ""))[:10]
        if not date:
            raise AdapterError(f"line {i + 1}: row carries no ts")
        if (repo, file) in by_key:
            skipped.append(f"{label}: duplicate row — first occurrence kept")
            continue
        env = make_env(
            device=device_name or versions.get("cpu") or device,
            runtime="litert-lm",
            runtime_version=runtime_version,
            soc=soc,
            os_build=os_build or versions.get("platform"),
            machine_label=machine_label,
        )
        record = _base_record("cpu", env, date)
        record["loads"] = True
        record["runs"] = True
        record["prefill_tokens_per_s"] = float(row["median_prefill_tps"])
        record["decode_tokens_per_s"] = float(row["median_decode_tps"])
        if row.get("median_ttft_s") is not None:
            record["ttft_ms"] = round(float(row["median_ttft_s"]) * 1000.0, 3)
        peaks = [float(inv.get("peak_rss_mb") or 0) for inv in invocations]
        peak = float(row.get("peak_rss_mb") or 0) or max(peaks)
        if peak > 0:
            record["peak_mem_mb"] = peak
        metrics: dict[str, Any] = {
            "prefill_tokens": float(row.get("prefill") or 0),
            "decode_tokens": float(row.get("decode") or 0),
            "init_s": float(row["median_init_s"]),
            "prefill_tps_min": float(row["min_prefill_tps"]),
            "prefill_tps_max": float(row["max_prefill_tps"]),
            "decode_tps_min": float(row["min_decode_tps"]),
            "decode_tps_max": float(row["max_decode_tps"]),
            "ttft_s_min": float(row["min_ttft_s"]),
            "ttft_s_max": float(row["max_ttft_s"]),
            "init_s_min": float(row["min_init_s"]),
            "init_s_max": float(row["max_init_s"]),
            "threads": float(row.get("threads") or 0),
            "invocations": float(len(invocations)),
            "runs_per_invocation": float(row.get("runs_per_invocation") or 0),
        }
        record["metrics"] = {k: v for k, v in metrics.items() if v != 0 or k in ("threads",)}
        record["evidence"] = [
            _PI5_LLM_CONDITIONS_NOTE,
            "versions: " + ", ".join(f"{k}={v}" for k, v in sorted(versions.items())),
            f"cache mode {row.get('cache')!r}; -p {row.get('prefill')} -d {row.get('decode')} "
            f"--runs {row.get('runs_per_invocation')} --cpu-thread-count {row.get('threads')}",
            f"gate ({gate.get('prompt')!r}): status {gate.get('status')}, exit {gate.get('exit')}, "
            f"wall {gate.get('wall_s')} s, output head {str(gate.get('output_head'))[:80]!r}",
            *(
                f"invocation {k}: exit={inv.get('exit')} wall_s={inv.get('wall_s')} "
                f"temp {inv.get('temp_before')}->{inv.get('temp_after')}C throttled={inv.get('throttled_after')} "
                f"prefill_tps={inv.get('prefill_tps')} decode_tps={inv.get('decode_tps')} "
                f"ttft_s={inv.get('ttft_s')} init_s={inv.get('init_s')} peak_rss_mb={inv.get('peak_rss_mb')}"
                for k, inv in enumerate(invocations)
            ),
        ]
        by_key[(repo, file)] = {"record": record, "date": date, "runtime_version": runtime_version}

    results: list[NpubenchResult] = []
    seen: set[str] = set()
    for (repo, file), e in sorted(by_key.items()):
        multi = len(repo_files.get(repo, set())) > 1
        name = file.rsplit("/", 1)[-1]
        tail = repo.split("/", 1)[-1]
        override = (model_id_for or {}).get(f"{tail}/{name}") or (model_id_for or {}).get(name)
        model_id = override or _pi5_llm_model_id(repo, file, multi)
        if model_id in seen:
            skipped.append(f"{repo}/{file}: derived id {model_id!r} collides with another row — refused")
            continue
        seen.add(model_id)
        results.append(NpubenchResult(model_id=model_id, artifact=name, date=e["date"], records=[e["record"]]))
    return results, skipped
