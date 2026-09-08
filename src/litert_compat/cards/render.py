"""CARD.md — rendered view of card.json.

Clean, standalone, dependency-free markdown (YAML front matter + tables): no
HTML, no site-specific includes, no relative asset links — it will be ingested
by RAG systems and AI crawlers. NOT a replacement for hand-written Hugging Face
cards: it is the structured superset their Performance tables derive from.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from litert_compat.cards.index import browser_status
from litert_compat.device_runs.records import device_status, record_condition

# The example-provenance banner. `edge-compat release-gate` checks that any
# card carrying example-provenance records renders this marker — keep the
# constant and the renderer in the same file so they cannot drift.
EXAMPLE_BANNER = (
    "> **Example data.** This card carries `example`-provenance records — "
    "pipeline fixtures, not real measurements."
)


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _metrics_cell(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "-"
    parts = [f"{k}={json.dumps(v)}" for k, v in sorted(metrics.items())]
    return " · ".join(parts)


def _harness_cell(record: dict[str, Any]) -> str:
    harness = record["harness"]
    if record.get("harness_version"):
        harness += f" {record['harness_version']}"
    statistic, runs = record.get("statistic"), record.get("runs")
    if statistic and runs:
        harness += f" ({statistic} of {runs})"
    elif statistic:
        harness += f" ({statistic})"
    return harness


def _bench_table(records: list[dict[str, Any]], with_runtime: bool) -> list[str]:
    has_metrics = any(r.get("metrics") for r in records)
    header = ["Device", "SoC", "Backend", "Precision", "Latency p50 (ms)", "Tokens/s",
              "Peak mem (MB)", "Harness", "Date", "Provenance"]
    if with_runtime:
        header.insert(0, "Runtime")
    if has_metrics:
        header.append("Additional metrics")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in records:
        row = [
            _cell(r["device"]), _cell(r.get("soc")), _cell(r["backend"]),
            _cell(r.get("precision")), _cell(r.get("latency_p50_ms")),
            _cell(r.get("tokens_per_s")), _cell(r.get("peak_mem_mb")),
            _harness_cell(r), _cell(r["date"]), _cell(r["provenance"]),
        ]
        if with_runtime:
            row.insert(0, _cell(r.get("runtime")))
        if has_metrics:
            row.append(_metrics_cell(r.get("metrics")))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def provenance_values(card: dict[str, Any]) -> set[str]:
    """Every provenance value the card presents (benchmarks, cross-runtime,
    delegation verdicts, browser and device records) — the release-gate's
    definition of 'carries example-provenance records'."""
    values = {r["provenance"] for r in card["benchmarks"]}
    values |= {r["provenance"] for r in card["cross_runtime"]}
    if card["delegation"] is not None:
        values |= set(card["delegation"]["matched_provenance_counts"])
    browser = card.get("browser")
    if browser is not None:
        values |= {r["provenance"] for r in browser["backends"]}
    device = card.get("device")
    if device is not None:
        values |= {entry["run"]["provenance"] for entry in device["records"]}
    return values


def _bool_cell(value: bool | None, true: str, false: str) -> str:
    if value is None:
        return "-"
    return true if value else false


def _env_cell(env: dict[str, Any]) -> str:
    browser = f"{env['browser']} {env['browser_version']}"
    if env["headless"]:
        browser += " headless"
    return (
        f"{env['machine_label']} · {browser} · {env['os']} · "
        f"@litertjs/core {env['litertjs_core_version']}"
    )


def _browser_section(browser: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Browser (LiteRT.js)",
        "",
        "Model-level sweep results — measured behavior of this exact `.tflite` in the "
        "browser, only meaningful together with the environment that produced it. "
        f"Full records (failure class, error evidence, sweep config): `{browser['sweep_source']}`.",
        "",
        "| Backend | Status | Full delegation | Output match | Max rel diff "
        "| Latency p50 (ms) | Environment | Date | Provenance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in browser["backends"]:
        lines.append(
            f"| {r['backend']} | {browser_status(r)} "
            f"| {_bool_cell(r['full_delegation'], 'yes', 'no')} "
            f"| {_bool_cell(r['output_match'], 'pass', 'MISMATCH')} "
            f"| {_cell(r['max_rel_diff'])} | {_cell(r['latency_p50_ms'])} "
            f"| {_env_cell(r['env'])} | {r['date']} | {r['provenance']} |"
        )
    if browser["demo_url"]:
        lines += ["", f"Live demo: {browser['demo_url']}"]
    return lines


def _device_env_cell(env: dict[str, Any]) -> str:
    parts = [env["device"]]
    if env["soc"]:
        parts.append(env["soc"])
    parts.append(f"{env['runtime']} {env['runtime_version']}")
    if env["vendor_sdk"]:
        parts.append(env["vendor_sdk"])
    if env["os_build"]:
        parts.append(env["os_build"])
    return " · ".join(parts)


def _device_section(device: dict[str, Any]) -> list[str]:
    sources = sorted({entry["source"] for entry in device["records"]})
    lines = [
        "",
        "## Device runs (NPU / LiteRT-LM)",
        "",
        "Model-level on-device results — measured behavior of this exact artifact on "
        "the recorded device, only meaningful together with that environment. "
        "Throughput is conditional on the measured prompt length (Prompt (tokens); "
        "'-' = the source stated none): rows differing only there are different "
        "measurements, not re-runs. Full "
        "records (failure classes, log evidence): "
        + ", ".join(f"`{s}`" for s in sources)
        + ".",
        "",
        "| Device | Accelerator | Status | Full delegation | Output match "
        "| Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) "
        "| Peak mem (MB) | Environment | Date | Provenance |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in device["records"]:
        run = entry["run"]
        condition = record_condition(run)
        signature = run.get("signature")
        if condition is not None:
            condition_cell = f"{condition:g}"
        elif signature is not None:
            # multi-signature .tflite: the measured signature is the condition
            condition_cell = f"sig:{signature}"
        else:
            condition_cell = "-"
        lines.append(
            f"| {entry['device']} | {run['accelerator']} | {device_status(run)} "
            f"| {_bool_cell(run['full_delegation'], 'yes', 'no')} "
            f"| {_bool_cell(run['output_match'], 'pass', 'MISMATCH')} "
            f"| {condition_cell} "
            f"| {_cell(run['latency_p50_ms'])} | {_cell(run['prefill_tokens_per_s'])} "
            f"| {_cell(run['decode_tokens_per_s'])} | {_cell(run['ttft_ms'])} "
            f"| {_cell(run['peak_mem_mb'])} | {_device_env_cell(run['env'])} "
            f"| {run['date']} | {run['provenance']} |"
        )
    return lines


def render_card_markdown(card: dict[str, Any]) -> str:
    model = card["model"]
    front_matter = yaml.safe_dump(
        {
            "model_id": model["id"],
            "family": model["family"],
            "task": model["task"],
            "license": model["license"],
            "source_url": model["source_url"],
        },
        sort_keys=True,
        default_flow_style=False,
    ).strip()

    lines = ["---", front_matter, "---", "", f"# {model['id']}", ""]

    if "example" in provenance_values(card):
        lines += [EXAMPLE_BANNER, ""]

    lines += [
        "| | |",
        "|---|---|",
        f"| **Task** | {model['task']} |",
        f"| **Family** | {model['family']} |",
        f"| **Source** | {model['source_url']} |",
        f"| **License** | {model['license']} |",
        "",
        "## Conversion",
        "",
        "| | |",
        "|---|---|",
        f"| **Tool** | {card['conversion']['tool']} {card['conversion']['tool_version']} |",
        f"| **Command** | `{card['conversion']['command']}` |",
        f"| **Quantization** | {card['conversion']['quantization']} |",
        "",
        "## Artifacts",
        "",
        "| File | SHA-256 | Size (MB) |",
        "|---|---|---|",
    ]
    for artifact in card["artifacts"]:
        lines.append(f"| `{artifact['file']}` | `{artifact['sha256']}` | {artifact['size_mb']} |")

    lines += ["", "## Performance", ""]
    if card["benchmarks"]:
        lines += _bench_table(card["benchmarks"], with_runtime=False)
    else:
        lines.append("No benchmark data yet.")

    delegation = card["delegation"]
    if delegation is not None:
        counts = delegation["matched_provenance_counts"]
        counts_cell = ", ".join(f"{k}={counts[k]}" for k in sorted(counts)) or "-"
        blocking_cell = ", ".join(delegation["blocking_ops"]) or "none"
        lines += [
            "",
            "## Delegation (static pre-flight)",
            "",
            "Static `edge-lint` verdicts against the delegate compatibility matrix — "
            "no runtime execution. Verdicts are only valid for this backend and "
            "LiteRT version.",
            "",
            "| | |",
            "|---|---|",
            f"| **Backend** | {delegation['backend']} |",
            f"| **LiteRT version** | {delegation['litert_version']} |",
            f"| **Op coverage** | {delegation['coverage_ops_pct']}% |",
            f"| **Partitions** | {delegation['partitions']} |",
            f"| **Blocking ops** | {blocking_cell} |",
            f"| **Verdict provenance** | {counts_cell} |",
            f"| **Lint report schema** | {delegation['lint_report_version']} |",
        ]

    browser = card.get("browser")
    if browser is not None:
        lines += _browser_section(browser)

    device = card.get("device")
    if device is not None:
        lines += _device_section(device)

    if card["cross_runtime"]:
        lines += ["", "## Cross-runtime", ""]
        lines += _bench_table(card["cross_runtime"], with_runtime=True)

    if card["pitfalls"]:
        lines += ["", "## Pitfalls", ""]
        lines += [f"- {pitfall}" for pitfall in card["pitfalls"]]

    lines += [
        "",
        "---",
        "",
        "_Rendered by `edge-card` from `card.json` (the source of truth); "
        "edit the inputs, not this file._",
    ]
    return "\n".join(lines) + "\n"
