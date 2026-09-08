"""`edge-compat device-run` — Phase 13 device-run ingestion and validation.

Exit-code contract (stable):

- `device-run validate`:            0 valid (and, with --cards, every card's
                                    device block matches the snapshots) ·
                                    1 findings · 2 usage error
- `device-run ingest-gpu-audit`:    0 written · 2 usage/data error
- `device-run ingest-devicemark`:   0 all rows ingested or expected-skipped ·
                                    1 refusals · 2 usage/data error
- `device-run ingest-compat-check`: 0 all ingested · 1 refusals · 2 usage/data error

Hard rules: these commands write ONLY under the given --out snapshot root —
never under data/matrix/ (the trap rule, device edition; tested) — and every
field a source does not carry is a caller-supplied option or an explicit
null, never an inferred value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from litert_compat.cards.enrich import device_coverage_findings
from litert_compat.cards.index import CardIndexError, collect_cards
from litert_compat.device_runs.adapters import (
    AdapterError,
    compat_check_records,
    console_record,
    devicemark_records,
    gpu_audit_record,
    make_env,
    npubench_parity_records,
    npubench_records,
    parse_console_log,
    pi5_benchmark_records,
    pi5_llm_records,
    parse_gpu_audit_log,
    parse_yardstick,
    yardstick_date,
    yardstick_record,
)
from litert_compat.device_runs.records import (
    DeviceRunError,
    device_run_errors,
    discover_device_run_snapshots,
    write_device_run,
)
from litert_compat.matrix.canonical import load_json

device_run_app = typer.Typer(
    no_args_is_help=True,
    help="Device-run results (Phase 13): NPU and LiteRT-LM lanes.",
)

OutOpt = Annotated[
    Path,
    typer.Option(
        "--out",
        file_okay=False,
        help="Device-run snapshot root; records land at <out>/<runtime_version>/<date>/.",
    ),
]
ArtifactShaOpt = Annotated[
    str | None,
    typer.Option(
        "--artifact-sha256",
        help="sha256 of the exact bytes measured, VERBATIM from a primary source (the "
        "ship's upload log, ship_sha256.txt, the curated manifest, or the hosting API read "
        "on the ship day) — never hand-computed from a file that may not be the one the log "
        "ran. Omit when no source states it (DECISIONS #167).",
    ),
]

DateOpt = Annotated[
    str,
    typer.Option(
        "--date",
        help="Measurement date (YYYY-MM-DD). The source does not stamp one; the owner "
        "supplies it — file mtimes are never used.",
    ),
]


def _write_or_exit(out: Path, doc: dict[str, Any], date: str) -> Path:
    try:
        return write_device_run(out, doc, date=date)
    except DeviceRunError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


@device_run_app.command()
def validate(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="One device-run result JSON file, or a snapshot root to discover.",
        ),
    ],
    cards_dir: Annotated[
        Path | None,
        typer.Option(
            "--cards",
            exists=True,
            file_okay=False,
            help="Also check every card's device block against the snapshots under PATH: "
            "each measured cell present, from its newest snapshot, verbatim (DECISIONS "
            "#165). PATH must be a snapshot root.",
        ),
    ] = None,
) -> None:
    """Validate one result file, or every snapshot under a root directory — and, with
    --cards, that the cards are exactly what enriching from that root would write."""
    if path.is_file():
        if cards_dir is not None:
            typer.echo("error: --cards needs a snapshot root, not a single file", err=True)
            raise typer.Exit(2)
        try:
            doc = load_json(path)
        except ValueError as exc:
            typer.echo(f"error: {path}: not valid JSON: {exc}", err=True)
            raise typer.Exit(2) from exc
        errors = device_run_errors(doc)
        if errors:
            typer.echo(f"{path}: INVALID")
            for error in errors:
                typer.echo(f"  {error}")
            raise typer.Exit(1)
        typer.echo(f"{path}: valid ({len(doc['results'])} record(s))")
        return
    try:
        snapshots = discover_device_run_snapshots(path)
    except DeviceRunError as exc:
        typer.echo(f"{path}: INVALID\n  {exc}")
        raise typer.Exit(1) from exc
    for snapshot in snapshots:
        typer.echo(
            f"{snapshot.path}: {snapshot.runtime} {snapshot.runtime_version} "
            f"@ {snapshot.date} — {len(snapshot.docs)} file(s)"
        )
    typer.echo(f"{len(snapshots)} snapshot(s), all valid")
    if cards_dir is None:
        return
    try:
        cards = collect_cards(cards_dir)
    except CardIndexError as exc:
        typer.echo(f"{cards_dir}: INVALID")
        for error in exc.errors:
            typer.echo(f"  {error}")
        raise typer.Exit(1) from exc
    try:
        findings = device_coverage_findings(cards, [path], cards_dir.resolve().parent)
    except DeviceRunError as exc:
        typer.echo(f"{path}: INVALID\n  {exc}")
        raise typer.Exit(1) from exc
    if findings:
        typer.echo(f"{cards_dir}: {len(findings)} device-block finding(s) against {path}")
        for finding in findings:
            typer.echo(f"  {finding}")
        raise typer.Exit(1)
    typer.echo(f"{cards_dir}: {len(cards)} card(s) match the snapshots under {path}")

@device_run_app.command("ingest-gpu-audit")
def ingest_gpu_audit(
    log_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="One gpu_gate_mac.sh *.gpu.log."),
    ],
    model_id: Annotated[str, typer.Option("--model-id", help="Catalog model id.")],
    device: Annotated[
        str, typer.Option("--device", help="Device slug for the file name (e.g. mac-m4-max).")
    ],
    runtime_version: Annotated[
        str,
        typer.Option(
            "--runtime-version",
            help="litert-lm version behind this log. Not stamped in the log; recover it "
            "(git history / lockfiles) rather than guess (owner decision #2).",
        ),
    ],
    date: DateOpt,
    device_name: Annotated[
        str | None,
        typer.Option("--device-name", help="Human-readable device name; defaults to the slug."),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    vendor_sdk: Annotated[str | None, typer.Option("--vendor-sdk")] = None,
    os_build: Annotated[str | None, typer.Option("--os-build")] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    quantization: Annotated[
        str | None,
        typer.Option(
            "--quantization",
            help="Quant variant, if the owner states it. Never inferred from file names.",
        ),
    ] = None,
    speeds_contaminated: Annotated[
        bool,
        typer.Option(
            "--speeds-contaminated",
            help="This log belongs to a batch whose speed figures the owner retracted "
            "(2026-07-22 contamination): the PASS/FAIL verdict is ingested, every "
            "throughput/latency figure is withheld (owner decision #5).",
        ),
    ] = False,
    artifact_sha256: ArtifactShaOpt = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest one gpu_audit log into a device-run record (LiteRT-LM Mac GPU lane).

    The verdict comes from the log body only: the CLI's exit code is untrusted
    (it is 0 even when engine creation fails) and summary.txt rows are not 1:1
    with surviving logs. Zero throughput in a results block is a failure.
    """
    try:
        parsed = parse_gpu_audit_log(log_file.read_text(encoding="utf-8"))
        record = gpu_audit_record(
            parsed,
            env=make_env(
                device=device_name or device,
                runtime="litert-lm",
                runtime_version=runtime_version,
                soc=soc,
                vendor_sdk=vendor_sdk,
                os_build=os_build,
                machine_label=machine_label,
            ),
            date=date,
            speeds_contaminated=speeds_contaminated,
        )
    except AdapterError as exc:
        typer.echo(f"error: {log_file}: {exc}", err=True)
        raise typer.Exit(2) from exc
    doc = {
        "schema_version": "1.0",
        "model_id": model_id,
        "device": device,
        "artifact": parsed.artifact,
        "quantization": quantization,
        "results": [record],
    }
    if artifact_sha256:
        doc["artifact_sha256"] = artifact_sha256
    target = _write_or_exit(out, doc, date)
    status = record["failure_class"] or ("pass" if record["runs"] else "unknown")
    typer.echo(f"{target}: {model_id} on {device} [{record['accelerator']}] — {status}")
    if parsed.unsupported_ops:
        ops = ", ".join(sorted(set(parsed.unsupported_ops)))
        typer.echo(
            f"note: log names unsupported op(s): {ops} — log-backed op-level matrix "
            "evidence, held back until the litert-lm version/date recovery lands "
            "(owner decision #2). This tool writes no matrix rows.",
            err=True,
        )


@device_run_app.command("ingest-console-log")
def ingest_console_log(
    log_file: Annotated[Path, typer.Argument(help="Raw runtime console log.")],
    model_id: Annotated[str, typer.Option("--model-id", help="Catalog model id.")],
    device: Annotated[
        str, typer.Option("--device", help="Device slug for the file name (e.g. pixel-8a).")
    ],
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            help="Accelerator this run used (gpu / cpu / npu). The log does not state "
            "it — it is the flag the runtime was invoked with, so the caller supplies it.",
        ),
    ],
    artifact: Annotated[
        str,
        typer.Option(
            "--artifact",
            help="Artifact file that was run. Not stamped in the log (litert_lm_main "
            "echoes no model path), so it is supplied rather than guessed from a name.",
        ),
    ],
    runtime_version: Annotated[
        str, typer.Option("--runtime-version", help="litert-lm version behind this log.")
    ],
    date: DateOpt,
    delegate_tag: Annotated[
        str,
        typer.Option(
            "--delegate-tag",
            help="Delegate whose 'Replacing N out of M' lines count as delegated. "
            "LITERT_CL for the Android OpenCL lane; a partially delegated graph also "
            "prints TfLiteXNNPackDelegate lines for the CPU remainder, and counting "
            "both would invent residency.",
        ),
    ] = "LITERT_CL",
    device_name: Annotated[
        str | None,
        typer.Option("--device-name", help="Human-readable device name; defaults to the slug."),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    os_build: Annotated[str | None, typer.Option("--os-build")] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    quantization: Annotated[
        str | None,
        typer.Option("--quantization", help="Quant variant, if the owner states it."),
    ] = None,
    prefill_tokens: Annotated[
        int | None,
        typer.Option(
            "--prefill-tokens",
            help="Prompt-length condition, for a log whose 'Number of tokens in "
            "prefill' header was cut before capture (a tail -N that kept only the "
            "results block). Supply it ONLY from evidence recorded with the run — "
            "the benchmark command line traced in the same log, say — never from "
            "memory or from a sibling run. A count that contradicts one the log "
            "does state is refused.",
        ),
    ] = None,
    decode_tokens: Annotated[
        int | None,
        typer.Option(
            "--decode-tokens",
            help="Decode-length counterpart of --prefill-tokens, under the same "
            "rule: supplied only from evidence recorded with the run.",
        ),
    ] = None,
    artifact_sha256: ArtifactShaOpt = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest a `litert_lm_main` / `litert-lm benchmark` console log.

    The Android lane runs the runtime binary directly, so unlike gpu_gate_mac.sh
    there is no wrapper header naming the artifact or the backend; both are
    caller-supplied here.
    """
    try:
        parsed = parse_console_log(
            log_file.read_text(encoding="utf-8", errors="replace"),
            delegate_tag=delegate_tag,
        )
        record = console_record(
            parsed,
            accelerator=accelerator,
            env=make_env(
                device=device_name or device,
                runtime="litert-lm",
                runtime_version=runtime_version,
                soc=soc,
                os_build=os_build,
                machine_label=machine_label,
            ),
            date=date,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
    except AdapterError as exc:
        typer.echo(f"error: {log_file}: {exc}", err=True)
        raise typer.Exit(2) from exc
    doc = {
        "schema_version": "1.0",
        "model_id": model_id,
        "device": device,
        "artifact": artifact,
        "quantization": quantization,
        "results": [record],
    }
    if artifact_sha256:
        doc["artifact_sha256"] = artifact_sha256
    target = _write_or_exit(out, doc, date)
    status = record["failure_class"] or ("pass" if record["runs"] else "unknown")
    delegation = (
        f", {record['delegated_ops']}/{record['total_ops']} delegated"
        if record["delegated_ops"] is not None
        else ""
    )
    typer.echo(f"{target}: {model_id} on {device} [{accelerator}] — {status}{delegation}")
    if parsed.unsupported_ops:
        ops = ", ".join(sorted(set(parsed.unsupported_ops)))
        typer.echo(
            f"note: log names unsupported op(s): {ops} — log-backed op-level matrix "
            "evidence. This tool writes no matrix rows; stage them for owner review.",
            err=True,
        )


@device_run_app.command("ingest-devicemark")
def ingest_devicemark(
    jsonl_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="devicemark measurements JSONL."),
    ],
    runtime_version: Annotated[
        str,
        typer.Option(
            "--runtime-version",
            help="litert-lm version behind these rows (not carried in the rows).",
        ),
    ],
    date: DateOpt,
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            help="Accelerator the leaderboard rows were measured on (not carried in the rows).",
        ),
    ],
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest devicemark leaderboard rows (LiteRT-LM lane; decode throughput).

    Only `*__litertlm` rows are this lane — `coreai` rows are skipped with a
    note (they belong in a card's cross_runtime list). Estimated memory
    (mem_measured: false) is carried as metrics.peak_mem_est_mb, never as a
    measured peak_mem_mb.
    """
    try:
        results, skipped = devicemark_records(
            jsonl_file.read_text(encoding="utf-8"),
            runtime_version=runtime_version,
            date=date,
            accelerator=accelerator,
            machine_label=machine_label,
        )
    except AdapterError as exc:
        typer.echo(f"error: {jsonl_file}: {exc}", err=True)
        raise typer.Exit(2) from exc
    for result in results:
        doc = {
            "schema_version": "1.0",
            "model_id": result.model_id,
            "device": result.device,
            "artifact": result.artifact,
            "quantization": result.quantization,
            "results": [result.record],
        }
        target = _write_or_exit(out, doc, date)
        typer.echo(
            f"{target}: {result.model_id} on {result.device} [{accelerator}] — "
            f"decode {result.record['decode_tokens_per_s']:g} tok/s"
        )
    for note in skipped:
        typer.echo(f"skipped: {note}", err=True)
    typer.echo(f"ingested {len(results)} row(s), skipped {len(skipped)}")


@device_run_app.command("ingest-npubench")
def ingest_npubench(
    results_jsonl: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False,
            help="s2_npu_sweep results.jsonl journal (classic .tflite NPU/GPU sweep).",
        ),
    ],
    runtime_version: Annotated[
        str,
        typer.Option(
            "--runtime-version",
            help="LiteRT (AAR) version inside the benchmark APK (not carried in the rows).",
        ),
    ],
    device: Annotated[
        str, typer.Option("--device", help="Device slug (e.g. galaxy-s26).")
    ],
    device_name: Annotated[
        str, typer.Option("--device-name", help="Human-readable device name.")
    ],
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    os_build: Annotated[str | None, typer.Option("--os-build")] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    repo_for: Annotated[
        list[str] | None,
        typer.Option(
            "--repo-for",
            help="<file>=<repo> for journals whose rows carry no repo/slug (the Pixel 8a "
            "run_p8a.py shape); repeatable; taken from the sweep's own README. A row that "
            "names a different repo for that file is refused (DECISIONS #167(d)).",
        ),
    ] = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest a classic-.tflite NPU/GPU sweep journal (npubench, JIT + AOT).

    Measured records come only from thermal-gated accepted rows (NPU rows also
    delegate-evidenced); failures keep their honest class; annulled windows,
    soc-mismatch skips and unobtainable files produce no record. JIT and AOT
    NPU rows are labeled via metrics.mode and env.vendor_sdk.
    """
    try:
        results, skipped = npubench_records(
            results_jsonl.read_text(encoding="utf-8"),
            runtime_version=runtime_version,
            device=device,
            device_name=device_name,
            soc=soc,
            os_build=os_build,
            machine_label=machine_label,
            repo_for=_parse_pairs(repo_for, "--repo-for"),
        )
    except AdapterError as exc:
        typer.echo(f"error: {results_jsonl}: {exc}", err=True)
        raise typer.Exit(2) from exc
    for result in results:
        doc = {
            "schema_version": "1.0",
            "model_id": result.model_id,
            "device": device,
            "artifact": result.artifact,
            "quantization": None,
            "results": result.records,
        }
        target = _write_or_exit(out, doc, result.date)
        bits = []
        for record in result.records:
            if record["failure_class"]:
                bits.append(f"{record['accelerator']}:{record['failure_class']}")
            else:
                bits.append(f"{record['accelerator']}:{record['latency_p50_ms']:g}ms")
        typer.echo(f"{target}: {result.model_id} — {', '.join(bits)}")
    for note in skipped:
        typer.echo(f"skipped: {note}", err=True)
    typer.echo(f"ingested {len(results)} artifact(s), skipped {len(skipped)}")
    if skipped and any("refusing to classify" in s for s in skipped):
        raise typer.Exit(1)


@device_run_app.command("ingest-compat-check")
def ingest_compat_check(
    report: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="compat_check JSON report."),
    ],
    date: DateOpt,
    device: Annotated[
        str, typer.Option("--device", help="Device slug the check ran on (e.g. mac-m4-max).")
    ],
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            help="Accelerator compat_check exercised (not carried in the report).",
        ),
    ],
    device_name: Annotated[
        str | None,
        typer.Option("--device-name", help="Human-readable device name; defaults to the slug."),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    vendor_sdk: Annotated[str | None, typer.Option("--vendor-sdk")] = None,
    os_build: Annotated[str | None, typer.Option("--os-build")] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    model_id_for: Annotated[
        list[str] | None,
        typer.Option(
            "--model-id-for",
            help="<repo tail>/<file>=<model id> (or <file>=<id>): file this result under the "
            "given id instead of the mechanical one — for continuity when a report carries "
            "only one of a repo's artifacts (DECISIONS #168); repeatable.",
        ),
    ] = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest a compat_check report (litert-lm version x artifact loadability).

    Refuse-until-sampled: only the `ok` status on the `llm` lane has a staged
    real sample; other branches are refused by name until the owner supplies
    one (owner decision #3). This lane stays a device-run input — the
    runtime-version axis is NOT absorbed into the matrix (spec §F.2).
    """
    try:
        doc = load_json(report)
    except ValueError as exc:
        typer.echo(f"error: {report}: not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(doc, dict):
        typer.echo(f"error: {report}: expected a JSON object", err=True)
        raise typer.Exit(2)
    try:
        runtime_version, results, refused = compat_check_records(
            doc,
            date=date,
            device=device_name or device,
            accelerator=accelerator,
            soc=soc,
            vendor_sdk=vendor_sdk,
            os_build=os_build,
            machine_label=machine_label,
            model_id_for=_parse_pairs(model_id_for, "--model-id-for") or None,
        )
    except AdapterError as exc:
        typer.echo(f"error: {report}: {exc}", err=True)
        raise typer.Exit(2) from exc
    for result in results:
        out_doc = {
            "schema_version": "1.0",
            "model_id": result.model_id,
            "device": device,
            "artifact": result.artifact,
            "quantization": None,
            "results": [result.record],
        }
        target = _write_or_exit(out, out_doc, date)
        typer.echo(f"{target}: {result.model_id} on {device} [{accelerator}] — ok")
    for note in refused:
        typer.echo(f"refused: {note}", err=True)
    typer.echo(
        f"ingested {len(results)} of {len(results) + len(refused)} result(s) "
        f"(litert-lm {runtime_version})"
    )
    if refused:
        raise typer.Exit(1)


if __name__ == "__main__":
    device_run_app()


@device_run_app.command("ingest-yardstick")
def ingest_yardstick(
    result_files: Annotated[
        list[Path],
        typer.Argument(help="One or more iOS BenchmarkApp result JSONs for ONE (model, device, "
                            "accelerator) — e.g. the quality run plus the vision probes; they "
                            "merge into one record."),
    ],
    model_id: Annotated[str, typer.Option("--model-id", help="Catalog model id.")],
    device: Annotated[
        str, typer.Option("--device", help="Device slug for the file name (e.g. iphone-17-pro).")
    ],
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            help="Accelerator this run used. NOT in the JSON: it is the app's --litert-cpu "
            "flag (absent = the runtime default, Metal GPU on iOS), so the caller supplies it.",
        ),
    ],
    artifact: Annotated[
        str,
        typer.Option(
            "--artifact",
            help="Published artifact file that was staged as the app's model.litertlm. The "
            "JSON only carries the staged name and the app catalog's (possibly stale) text.",
        ),
    ],
    runtime_version: Annotated[
        str, typer.Option("--runtime-version", help="litert-lm version linked into the app build.")
    ],
    date: Annotated[
        str | None,
        typer.Option("--date", help="Override the measurement date; default = the UTC calendar "
                                    "day of the source timestamps (they are stamped)."),
    ] = None,
    device_name: Annotated[
        str | None,
        typer.Option("--device-name", help="Human-readable device name; defaults to the slug."),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    quantization: Annotated[
        str | None,
        typer.Option("--quantization", help="Quant variant, if the owner states it."),
    ] = None,
    artifact_sha256: ArtifactShaOpt = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest iOS BenchmarkApp (`--yardstick-autorun`) result JSONs.

    The JSON stamps its own UTC timestamp, task id, device model/OS and the
    measured metrics; it does not carry the runtime version, the accelerator,
    or the published artifact name — those are caller-supplied.
    """
    try:
        runs = [parse_yardstick(json.loads(p.read_text(encoding="utf-8"))) for p in result_files]
        measured_date = date or yardstick_date(runs)
        os_builds = sorted({f"iOS {r.os_version}" for r in runs if r.os_version})
        models = sorted({r.device_model for r in runs if r.device_model})
        record = yardstick_record(
            runs,
            accelerator=accelerator,
            env=make_env(
                device=device_name or device,
                runtime="litert-lm",
                runtime_version=runtime_version,
                soc=soc,
                os_build=", ".join(os_builds) or None,
                machine_label=machine_label,
            ),
            date=measured_date,
        )
        if models:
            record["evidence"].insert(0, f"device.modelIdentifier={'/'.join(models)}")
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    doc = {
        "schema_version": "1.0",
        "model_id": model_id,
        "device": device,
        "artifact": artifact,
        "quantization": quantization,
        "results": [record],
    }
    if artifact_sha256:
        doc["artifact_sha256"] = artifact_sha256
    target = _write_or_exit(out, doc, measured_date)
    typer.echo(
        f"{target}: {model_id} on {device} [{accelerator}] — pass "
        f"({len(runs)} task(s) merged: {', '.join(r.task for r in runs)})"
    )


def _parse_pairs(values: list[str] | None, flag: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            typer.echo(f"error: {flag} expects <key>=<value>, got {value!r}", err=True)
            raise typer.Exit(2)
        key, _, mapped = value.partition("=")
        pairs[key.strip()] = mapped.strip()
    return pairs


@device_run_app.command("ingest-pi5-benchmark")
def ingest_pi5_benchmark(
    results_jsonl: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False,
            help="pi5_bench.py results.jsonl (Raspberry Pi 5 classic .tflite sweep, benchmark_model).",
        ),
    ],
    device: Annotated[str, typer.Option("--device", help="Device slug (e.g. raspberry-pi-5).")],
    device_name: Annotated[
        str | None,
        typer.Option("--device-name", help="Human-readable device name; defaults to the row's versions.cpu."),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    os_build: Annotated[
        str | None,
        typer.Option("--os-build", help="Defaults to the row's versions.platform (kernel + glibc)."),
    ] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest a Raspberry Pi 5 `benchmark_model` sweep journal (CPU/XNNPACK).

    The runtime version is read from each row's own `versions` (the
    ai-edge-litert nightly the binary links), so a nightly snapshot directory
    reads e.g. 2.2.0.dev20260804. One record per measured signature, carried in
    the record's `signature`; rows not accepted by the driver's own rule (exit,
    XNNPACK, throttle word) or carrying errors are skipped, never guessed.
    """
    try:
        results, skipped = pi5_benchmark_records(
            results_jsonl.read_text(encoding="utf-8"),
            device=device, device_name=device_name, soc=soc, os_build=os_build,
            machine_label=machine_label,
        )
    except AdapterError as exc:
        typer.echo(f"error: {results_jsonl}: {exc}", err=True)
        raise typer.Exit(2) from exc
    for result in results:
        doc = {
            "schema_version": "1.0",
            "model_id": result.model_id,
            "device": device,
            "artifact": result.artifact,
            "quantization": None,
            "results": result.records,
        }
        target = _write_or_exit(out, doc, result.date)
        cells = ", ".join(
            f"{r['accelerator']}{'@' + r['signature'] if r.get('signature') else ''}:"
            f"{r['latency_p50_ms']:g}ms"
            for r in result.records
        )
        typer.echo(f"{target}: {result.model_id} — {cells}")
    for note in skipped:
        typer.echo(f"skipped: {note}", err=True)
    typer.echo(f"ingested {len(results)} artifact(s), skipped {len(skipped)}")


@device_run_app.command("ingest-npubench-parity")
def ingest_npubench_parity(
    report: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="s26_gate.py parity report (keyed JSON)."),
    ],
    repo: Annotated[str, typer.Option("--repo", help="Hub repo the report's variants belong to.")],
    file_for: Annotated[
        list[str],
        typer.Option("--file", help="<variant>=<published .tflite file>; repeatable, one per report variant."),
    ],
    runtime_version: Annotated[
        str, typer.Option("--runtime-version", help="LiteRT (AAR) version inside the benchmark APK."),
    ],
    date: DateOpt,
    device: Annotated[str, typer.Option("--device", help="Device slug (e.g. galaxy-s26).")],
    device_name: Annotated[str, typer.Option("--device-name", help="Human-readable device name.")],
    signature_for: Annotated[
        list[str] | None,
        typer.Option(
            "--signature",
            help="<dur>=<signature name> from the logcat's asset suffix (e.g. 10s=transcribe_10s); "
            "unmapped durations keep the raw key token.",
        ),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    os_build: Annotated[str | None, typer.Option("--os-build")] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest an npubench `#parity` gate report (keyed JSON, on-device parity).

    The report names no files, signatures, or timestamps: the caller maps them
    from the lane's own scripts and logcat. Parity figures land in metrics;
    output_match stays null (the report's own criterion, not the shared
    tolerance policy).
    """
    try:
        doc_in = json.loads(report.read_text(encoding="utf-8"))
        results, skipped = npubench_parity_records(
            doc_in, repo=repo, files=_parse_pairs(file_for, "--file"),
            signature_for=_parse_pairs(signature_for, "--signature"),
            runtime_version=runtime_version, date=date, device=device,
            device_name=device_name, soc=soc, os_build=os_build, machine_label=machine_label,
        )
    except (AdapterError, ValueError) as exc:
        typer.echo(f"error: {report}: {exc}", err=True)
        raise typer.Exit(2) from exc
    for result in results:
        doc = {
            "schema_version": "1.0",
            "model_id": result.model_id,
            "device": device,
            "artifact": result.artifact,
            "quantization": None,
            "results": result.records,
        }
        target = _write_or_exit(out, doc, date)
        cells = ", ".join(
            f"{r['accelerator']}@{r.get('signature')}:"
            + (f"{r['latency_p50_ms']:g}ms" if r["latency_p50_ms"] is not None else str(r["failure_class"]))
            for r in result.records
        )
        typer.echo(f"{target}: {result.model_id} — {cells}")
    for note in skipped:
        typer.echo(f"skipped: {note}", err=True)
    typer.echo(f"ingested {len(results)} artifact(s), skipped {len(skipped)}")


@device_run_app.command("ingest-pi5-llm")
def ingest_pi5_llm(
    results_jsonl: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False,
            help="pi5_llm_bench.py llm_results.jsonl (Raspberry Pi 5 LLM .litertlm sweep, litert-lm benchmark).",
        ),
    ],
    device: Annotated[str, typer.Option("--device", help="Device slug (e.g. raspberry-pi-5).")],
    device_name: Annotated[
        str | None,
        typer.Option("--device-name", help="Human-readable device name; defaults to the row's versions.cpu."),
    ] = None,
    soc: Annotated[str | None, typer.Option("--soc")] = None,
    os_build: Annotated[
        str | None,
        typer.Option("--os-build", help="Defaults to the row's versions.platform (kernel + glibc)."),
    ] = None,
    machine_label: Annotated[str | None, typer.Option("--machine-label")] = None,
    model_id_for: Annotated[
        list[str] | None,
        typer.Option(
            "--model-id-for",
            help="<repo tail>/<file>=<model id> (or <file>=<id>) for bundles whose catalog id is the "
            "cards' hand-chosen one rather than the mechanical repo-derived id; repeatable. Joins "
            "across lanes stay a rendering concern — this only names the id the row is filed under.",
        ),
    ] = None,
    out: OutOpt = Path("data/device_runs"),
) -> None:
    """Ingest a Raspberry Pi 5 LLM sweep journal (`litert-lm benchmark`, CPU).

    The runtime version is read from each row's own `versions["litert-lm"]`, the
    date from its `ts`; throughput is the row's median over three invocations with
    the spread kept in `metrics`; rows whose own generation gate did not pass, or
    whose invocations exited non-zero / throttled, are skipped, never guessed.
    """
    try:
        results, skipped = pi5_llm_records(
            results_jsonl.read_text(encoding="utf-8"),
            device=device, device_name=device_name, soc=soc, os_build=os_build,
            machine_label=machine_label,
            model_id_for=_parse_pairs(model_id_for, "--model-id-for") or None,
        )
    except AdapterError as exc:
        typer.echo(f"error: {results_jsonl}: {exc}", err=True)
        raise typer.Exit(2) from exc
    for result in results:
        doc = {
            "schema_version": "1.0",
            "model_id": result.model_id,
            "device": device,
            "artifact": result.artifact,
            "quantization": None,
            "results": result.records,
        }
        target = _write_or_exit(out, doc, result.date)
        rec = result.records[0]
        typer.echo(
            f"{target}: {result.model_id} — cpu prefill {rec['prefill_tokens_per_s']:g} / "
            f"decode {rec['decode_tokens_per_s']:g} tok/s"
        )
    for note in skipped:
        typer.echo(f"skipped: {note}", err=True)
    typer.echo(f"ingested {len(results)} artifact(s), skipped {len(skipped)}")
