"""Phase 13 device-run lane: schema + example fixtures, strict snapshot
discovery, append-only writes, the three ingestion adapters (built against
the REAL staged samples in data/examples/llm_samples/ with the recorded owner
decisions enforced), the device-run CLI, card enrichment, and the per-runtime
freshness delta.

Committed fixtures are example-provenance; adapter tests parse the staged
real samples but write nothing outside tmp directories.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from helpers import example_doc, example_record, write_snapshot
from litert_compat.cards.cli import app as cards_app
from litert_compat.cards.enrich import load_device_runs
from litert_compat.cli import app as compat_app
from litert_compat.device_runs.adapters import (
    CONTAMINATED_NOTE,
    AdapterError,
    compat_check_records,
    device_slug,
    devicemark_records,
    gpu_audit_record,
    make_env,
    parse_gpu_audit_log,
)
from litert_compat.device_runs.records import (
    DeviceRunError,
    device_run_errors,
    device_status,
    discover_device_run_snapshots,
    record_label,
    snapshots_by_runtime,
    write_device_run,
)
from litert_compat.freshness.delta import (
    compute_device_delta,
    device_entry_heading,
    render_device_entry,
    upsert_entry,
)
from litert_compat.matrix.canonical import canonical_dumps, load_json

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
DEVICE_RUN_EXAMPLES = EXAMPLES / "device_runs"
LLM_SAMPLES = EXAMPLES / "llm_samples"

ENV = make_env(device="Test Mac", runtime="litert-lm", runtime_version="0.15.0")


# --- schema + committed fixtures --------------------------------------------


def test_committed_example_fixtures_validate() -> None:
    files = sorted(DEVICE_RUN_EXAMPLES.rglob("*.json"))
    assert len(files) == 2
    for path in files:
        doc = load_json(path)
        assert device_run_errors(doc) == [], path
        assert all(r["provenance"] == "example" for r in doc["results"])


def test_discover_committed_example_snapshots() -> None:
    snapshots = discover_device_run_snapshots(DEVICE_RUN_EXAMPLES)
    assert [(s.runtime, s.runtime_version, s.date) for s in snapshots] == [
        ("litert", "0.0.0-example", "2026-01-01"),
        ("litert-lm", "0.0.0-example", "2026-01-02"),
    ]
    groups = snapshots_by_runtime(snapshots)
    assert list(groups) == ["litert", "litert-lm"]


def test_schema_rejects_missing_env_and_bad_shapes() -> None:
    doc = load_json(
        DEVICE_RUN_EXAMPLES / "0.0.0-example" / "2026-01-01"
        / "example-tiny-clean__example-phone.json"
    )
    stripped = json.loads(json.dumps(doc))
    del stripped["results"][0]["env"]
    assert device_run_errors(stripped)  # the honesty rule is mechanical
    zero = json.loads(json.dumps(doc))
    zero["results"][0]["decode_tokens_per_s"] = 0.0
    assert device_run_errors(zero)  # a printed 0.00 is a failure marker, not a value


def test_device_status_vocabulary() -> None:
    assert device_status(example_record()) == "pass"
    assert device_status(example_record(loads=False, runs=False)) == "load_failed"
    assert device_status(example_record(runs=False)) == "run_failed"
    assert device_status(example_record(output_match=False)) == "output_mismatch"
    assert device_status(example_record(full_delegation=False)) == "fallback"


# --- snapshot discovery strictness ------------------------------------------


def test_discovery_rejects_version_mismatch(tmp_path: Path) -> None:
    doc = example_doc("m", "d", [example_record(runtime_version="9.9.9")])
    write_snapshot(tmp_path / "runs", "0.1.0", "2026-01-01", [doc])
    with pytest.raises(DeviceRunError, match="version axis"):
        discover_device_run_snapshots(tmp_path / "runs")


def test_discovery_rejects_mixed_runtimes(tmp_path: Path) -> None:
    docs = [
        example_doc("m1", "d", [example_record(runtime="litert-lm", runtime_version="0.1.0")]),
        example_doc(
            "m2", "d", [example_record("npu_qnn", runtime="litert", runtime_version="0.1.0")]
        ),
    ]
    write_snapshot(tmp_path / "runs", "0.1.0", "2026-01-01", docs)
    with pytest.raises(DeviceRunError, match="mixes runtimes"):
        discover_device_run_snapshots(tmp_path / "runs")


def test_discovery_rejects_bad_filename(tmp_path: Path) -> None:
    snap = tmp_path / "runs" / "0.0.0-example" / "2026-01-01"
    snap.mkdir(parents=True)
    (snap / "wrong.json").write_text(
        canonical_dumps(example_doc("m", "d", [example_record()])), encoding="utf-8"
    )
    with pytest.raises(DeviceRunError, match="file name"):
        discover_device_run_snapshots(tmp_path / "runs")


# --- append-only writes ------------------------------------------------------


def test_write_device_run_appends_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    doc = example_doc("m", "d", [example_record("gpu")])
    target = write_device_run(root, doc, date="2026-01-01")
    assert target == root / "0.0.0-example" / "2026-01-01" / "m__d.json"
    assert device_run_errors(load_json(target)) == []

    # A second accelerator merges in, sorted.
    more = example_doc("m", "d", [example_record("cpu", decode=10.0)])
    write_device_run(root, more, date="2026-01-01")
    merged = load_json(target)
    assert [r["accelerator"] for r in merged["results"]] == ["cpu", "gpu"]

    # Re-ingesting the identical record is a no-op (partial runs re-run
    # safely); the file's bytes stay untouched.
    before = target.read_bytes()
    assert write_device_run(root, doc, date="2026-01-01") == target
    assert target.read_bytes() == before
    # A DIFFERENT record on an already-taken accelerator is refused —
    # append-only, never overwritten.
    changed = example_doc("m", "d", [example_record("gpu", decode=99.0)])
    with pytest.raises(DeviceRunError, match="append-only"):
        write_device_run(root, changed, date="2026-01-01")
    # Disagreeing top-level facts are refused.
    conflicting = example_doc("m", "d", [example_record("npu")])
    conflicting["artifact"] = "other.litertlm"
    with pytest.raises(DeviceRunError, match="disagrees"):
        write_device_run(root, conflicting, date="2026-01-01")
    # An invalid document is never written.
    bad = example_doc("m", "d", [example_record("x")])
    del bad["results"][0]["env"]
    with pytest.raises(DeviceRunError, match="invalid"):
        write_device_run(root, bad, date="2026-01-01")


def test_write_device_run_deterministic(tmp_path: Path) -> None:
    doc = example_doc("m", "d", [example_record("gpu"), example_record("cpu", decode=1.0)])
    a = write_device_run(tmp_path / "a", doc, date="2026-01-01").read_bytes()
    b = write_device_run(tmp_path / "b", doc, date="2026-01-01").read_bytes()
    assert a == b


def test_write_device_run_prompt_length_is_part_of_record_identity(tmp_path: Path) -> None:
    """DECISIONS #162: the same accelerator under a different prompt length is
    a different fact — it merges in instead of tripping the append-only guard,
    and the file orders records accelerator-first, condition ascending."""
    root = tmp_path / "runs"
    floor = example_doc("m", "d", [example_record("gpu", metrics={"prefill_tokens": 19})])
    target = write_device_run(root, floor, date="2026-01-01")

    long_prompt = example_doc(
        "m", "d", [example_record("gpu", decode=35.0, metrics={"prefill_tokens": 205})]
    )
    write_device_run(root, long_prompt, date="2026-01-01")
    unconditioned = example_doc("m", "d", [example_record("gpu", decode=50.0)])
    write_device_run(root, unconditioned, date="2026-01-01")
    merged = load_json(target)
    assert [record_label(r) for r in merged["results"]] == [
        "gpu", "gpu@19tok", "gpu@205tok",
    ]

    # Same accelerator, same condition, different figures: still refused.
    changed = example_doc(
        "m", "d", [example_record("gpu", decode=99.0, metrics={"prefill_tokens": 205})]
    )
    with pytest.raises(DeviceRunError, match="gpu@205tok"):
        write_device_run(root, changed, date="2026-01-01")
    # Identical conditioned record: idempotent no-op.
    before = target.read_bytes()
    assert write_device_run(root, long_prompt, date="2026-01-01") == target
    assert target.read_bytes() == before


# --- gpu_audit adapter (lane 1, staged real samples) -------------------------


def test_gpu_audit_pass_shape() -> None:
    log = parse_gpu_audit_log(
        (LLM_SAMPLES / "gpu_audit" / "gemma3-1b-official.gpu.log").read_text()
    )
    assert log.artifact == "gemma3-1b-it-int4.litertlm"
    assert log.accelerator == "gpu"
    record = gpu_audit_record(log, env=ENV, date="2026-07-23")
    assert record["loads"] and record["runs"]
    assert record["prefill_tokens_per_s"] == 4343.84
    assert record["decode_tokens_per_s"] == 185.75
    assert record["ttft_ms"] == 64.3
    assert record["metrics"] == {
        "prefill_tokens": 256.0,
        "decode_tokens": 256.0,
        "max_num_tokens": 1024.0,
        "init_s": 1.8673,
    }
    assert record["full_delegation"] is None  # the log states nothing on PASS
    doc = example_doc("gemma3-1b", "test-mac", [record])
    doc["artifact"] = log.artifact
    assert device_run_errors(doc) == []


def test_gpu_audit_engine_create_failed_shape() -> None:
    """The FAIL log: verdict from the log body (the CLI exited 0 — untrusted),
    with the unsupported-op block captured as evidence and the partition
    counts the log states."""
    log = parse_gpu_audit_log(
        (LLM_SAMPLES / "gpu_audit" / "lfm25-12b-instruct-int4.gpu.log").read_text()
    )
    record = gpu_audit_record(log, env=ENV, date="2026-07-22")
    assert not record["loads"] and not record["runs"]
    assert record["failure_class"] == "engine_create_failed"
    assert record["full_delegation"] is False
    assert record["delegated_ops"] == 536
    assert record["total_ops"] == 579
    # The op-level evidence the owner will curate matrix rows from later:
    assert sorted(set(log.unsupported_ops)) == [
        "ADD", "CAST", "GATHER_ND", "GREATER_EQUAL", "LESS_EQUAL", "SUM",
    ]
    assert any("GATHER_ND: Operation is not supported." in e for e in record["evidence"])
    assert any("536 operations will run on the GPU" in e for e in record["evidence"])
    assert record["error"] and "Failed to create engine" in record["error"]


def test_gpu_audit_zero_throughput_is_a_failure() -> None:
    """A printed results block with 0.00 tokens/s did NOT run (the summary
    row says PASS — also untrusted)."""
    log = parse_gpu_audit_log(
        (LLM_SAMPLES / "gpu_audit" / "lfm25_wi8_composite_conv040dev_0150.gpu.log").read_text()
    )
    record = gpu_audit_record(log, env=ENV, date="2026-08-08")
    assert record["loads"] and not record["runs"]
    assert record["failure_class"] == "zero_throughput"
    assert record["prefill_tokens_per_s"] is None
    assert record["decode_tokens_per_s"] is None
    assert record["error"] and "Failed to generate content" in record["error"]
    assert record["metrics"] == {"init_s": 1.3646}  # init genuinely happened


def test_gpu_audit_contaminated_batch_withholds_speeds() -> None:
    """Owner decision #5: 2026-07-22 batch speeds are import-blocked; the
    PASS/FAIL verdict stands."""
    log = parse_gpu_audit_log(
        (LLM_SAMPLES / "gpu_audit" / "gemma3-1b-official.gpu.log").read_text()
    )
    record = gpu_audit_record(log, env=ENV, date="2026-07-22", speeds_contaminated=True)
    assert record["loads"] and record["runs"]  # verdict kept
    assert record["prefill_tokens_per_s"] is None
    assert record["decode_tokens_per_s"] is None
    assert record["ttft_ms"] is None
    assert "init_s" not in record.get("metrics", {})
    assert CONTAMINATED_NOTE in record["evidence"]


def test_gpu_audit_rejects_non_log_input() -> None:
    with pytest.raises(AdapterError, match="Benchmarking model"):
        parse_gpu_audit_log("not a log\n")


# --- devicemark adapter (lane 2) ---------------------------------------------


def test_devicemark_rows_split_lanes_and_memory_honestly() -> None:
    text = (LLM_SAMPLES / "devicemark" / "measurements.slice.jsonl").read_text()
    results, skipped = devicemark_records(
        text, runtime_version="0.15.0", date="2026-07-22", accelerator="cpu"
    )
    # Only the *__litertlm rows are this lane; coreai rows are cross_runtime.
    assert [(r.model_id, r.device) for r in results] == [
        ("gemma-4-e2b", "m4-max"),
        ("gemma-4-e2b", "iphone-17-pro"),
    ]
    assert len(skipped) == 3
    assert all("cross_runtime" in note for note in skipped)
    for result in results:
        assert result.quantization == "int4"
        assert result.record["decode_tokens_per_s"] > 0
        # These rows carry mem_measured: true -> the measured field is honest.
        assert result.record["peak_mem_mb"] == 488.0
        assert "PipelinedBench" in result.record["evidence"][0]
        doc = example_doc(result.model_id, result.device, [result.record])
        doc["artifact"] = result.artifact
        doc["quantization"] = result.quantization
        assert device_run_errors(doc) == []


def test_devicemark_estimated_memory_never_measured() -> None:
    """Owner decision #4: mem_measured false -> metrics.peak_mem_est_mb, never
    peak_mem_mb."""
    row = {
        "artifact_id": "some-model__int8__litertlm",
        "runtime": "litertlm",
        "device": "M4 Max",
        "decode_tok_s": 50,
        "peak_mem_mb": 1600,
        "mem_measured": False,
        "power_w": None,
    }
    results, skipped = devicemark_records(
        json.dumps(row), runtime_version="0.15.0", date="2026-07-22", accelerator="cpu"
    )
    assert skipped == []
    record = results[0].record
    assert record["peak_mem_mb"] is None
    assert record["metrics"] == {"peak_mem_est_mb": 1600.0}


def test_devicemark_refuses_unsampled_shapes() -> None:
    bad_artifact = '{"artifact_id": "no-format", "runtime": "litertlm", "device": "X"}'
    with pytest.raises(AdapterError, match="artifact_id"):
        devicemark_records(
            bad_artifact, runtime_version="0.15.0", date="2026-07-22", accelerator="cpu"
        )
    zero_decode = json.dumps(
        {
            "artifact_id": "m__int4__litertlm",
            "runtime": "litertlm",
            "device": "X",
            "decode_tok_s": 0,
        }
    )
    with pytest.raises(AdapterError, match="refusing to guess"):
        devicemark_records(
            zero_decode, runtime_version="0.15.0", date="2026-07-22", accelerator="cpu"
        )


def test_device_slug() -> None:
    assert device_slug("M4 Max") == "m4-max"
    assert device_slug("iPhone 17 Pro") == "iphone-17-pro"
    assert device_slug("Pixel 8a") == "pixel-8a"


# --- compat_check adapter (lane 3) -------------------------------------------


def test_compat_check_ok_branch_ingests() -> None:
    doc = load_json(LLM_SAMPLES / "compat_check" / "compat_0.15.0.json")
    version, results, refused = compat_check_records(
        doc, date="2026-08-10", device="Test Mac", accelerator="cpu"
    )
    assert version == "0.15.0"
    assert refused == []
    assert [r.model_id for r in results] == ["granite-4.0-h-350m"]
    record = results[0].record
    assert record["loads"] and record["runs"]
    assert record["output_match"] is None  # a sanity answer is not a numeric comparison
    assert "The answer to 17 + 25 is 42." in record["evidence"][0]
    out = example_doc(results[0].model_id, "test-mac", [record])
    out["artifact"] = results[0].artifact
    assert device_run_errors(out) == []


def test_compat_check_multi_artifact_repo_gets_suffixed_ids() -> None:
    """Real sample (compat_0.16.0.json): granite-4.0-h-350m ships fp16 AND
    int8 — the bare repo id collides on one record file per (model, device),
    so multi-artifact repos get per-artifact ids suffixed with the artifact
    stem remainder, like the console-log lane's quant-suffixed ids."""
    doc = {
        "runtime": "litert-lm", "runtime_version": "0.16.0",
        "results": [
            {"repo": "litert-community/granite-4.0-h-1b",
             "file": "granite-4.0-h-1b_int8.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
            {"repo": "litert-community/granite-4.0-h-350m",
             "file": "granite-4.0-h-350m_fp16.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
            {"repo": "litert-community/granite-4.0-h-350m",
             "file": "granite-4.0-h-350m_int8.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
        ],
    }
    _, results, refused = compat_check_records(
        doc, date="2026-08-12", device="Test Mac", accelerator="cpu"
    )
    assert refused == []
    assert [r.model_id for r in results] == [
        "granite-4.0-h-1b",  # single artifact keeps the bare id (continuity)
        "granite-4.0-h-350m-fp16",
        "granite-4.0-h-350m-int8",
    ]
    assert [r.artifact for r in results] == [
        "granite-4.0-h-1b_int8.litertlm",
        "granite-4.0-h-350m_fp16.litertlm",
        "granite-4.0-h-350m_int8.litertlm",
    ]


def test_compat_check_lowercases_cased_repo_names() -> None:
    """Real sample (compat_0.16.0.json): most litert-community repos are cased
    (LFM2.5-1.2B-Instruct, Qwen3.5-0.8B) — ids are derived by mechanical
    lowercasing only, with the multi-artifact suffix on top."""
    doc = {
        "runtime": "litert-lm", "runtime_version": "0.16.0",
        "results": [
            {"repo": "litert-community/LFM2.5-1.2B-Instruct",
             "file": "LFM2.5-1.2B-Instruct_int4.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
            {"repo": "litert-community/LFM2.5-1.2B-Instruct",
             "file": "LFM2.5-1.2B-Instruct_int8.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
            {"repo": "litert-community/Qwen3.5-0.8B",
             "file": "Qwen3.5-0.8B_int8.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
        ],
    }
    _, results, refused = compat_check_records(
        doc, date="2026-08-12", device="Test Mac", accelerator="cpu"
    )
    assert refused == []
    assert [r.model_id for r in results] == [
        "lfm2.5-1.2b-instruct-int4",
        "lfm2.5-1.2b-instruct-int8",
        "qwen3.5-0.8b",  # single artifact keeps the bare (lowercased) id
    ]


def test_compat_check_refuses_unsampled_branches() -> None:
    """Owner decision #3: refuse-until-sampled — no invented BROKEN/SUSPECT
    strings, no invented vlm/tflite lanes."""
    doc = {
        "runtime": "litert-lm", "runtime_version": "0.15.0",
        "results": [
            {"repo": "x/broken-model", "file": "b.litertlm", "lane": "llm",
             "status": "BROKEN", "error": "boom", "answer": None},
            {"repo": "x/vlm-model", "file": "v.litertlm", "lane": "vlm",
             "status": "ok", "error": None, "answer": "42"},
            {"repo": "x/ok-model", "file": "ok.litertlm", "lane": "llm",
             "status": "ok", "error": None, "answer": "42"},
        ],
    }
    _, results, refused = compat_check_records(
        doc, date="2026-08-10", device="Test Mac", accelerator="cpu"
    )
    assert [r.model_id for r in results] == ["ok-model"]
    assert len(refused) == 2
    assert any("'BROKEN' has no staged sample" in note for note in refused)
    assert any("lane 'vlm' has no staged sample" in note for note in refused)


# --- device-run CLI ----------------------------------------------------------


def test_cli_ingest_gpu_audit_and_validate(tmp_path: Path) -> None:
    out = tmp_path / "device_runs"
    result = runner.invoke(
        compat_app,
        [
            "device-run", "ingest-gpu-audit",
            str(LLM_SAMPLES / "gpu_audit" / "lfm25-12b-instruct-int4.gpu.log"),
            "--model-id", "lfm2.5-1.2b-instruct", "--device", "test-mac",
            "--device-name", "Test Mac", "--runtime-version", "0.15.0",
            "--date", "2026-07-22", "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "engine_create_failed" in result.stdout
    # The op-level evidence is surfaced as a note, and no matrix row is written.
    assert "unsupported op(s)" in result.stderr
    assert "GATHER_ND" in result.stderr
    target = out / "0.15.0" / "2026-07-22" / "lfm2.5-1.2b-instruct__test-mac.json"
    assert device_run_errors(load_json(target)) == []

    assert runner.invoke(compat_app, ["device-run", "validate", str(out)]).exit_code == 0
    assert runner.invoke(compat_app, ["device-run", "validate", str(target)]).exit_code == 0

    # Re-ingesting the same record is refused: snapshots are append-only.
    again = runner.invoke(
        compat_app,
        [
            "device-run", "ingest-gpu-audit",
            str(LLM_SAMPLES / "gpu_audit" / "lfm25-12b-instruct-int4.gpu.log"),
            "--model-id", "lfm2.5-1.2b-instruct", "--device", "test-mac",
            "--runtime-version", "0.15.0", "--date", "2026-07-22", "--out", str(out),
        ],
    )
    assert again.exit_code == 2
    assert "append-only" in again.stderr


def test_cli_ingest_devicemark(tmp_path: Path) -> None:
    out = tmp_path / "device_runs"
    result = runner.invoke(
        compat_app,
        [
            "device-run", "ingest-devicemark",
            str(LLM_SAMPLES / "devicemark" / "measurements.slice.jsonl"),
            "--runtime-version", "0.15.0", "--date", "2026-07-22",
            "--accelerator", "cpu", "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ingested 2 row(s), skipped 3" in result.stdout
    files = sorted(p.name for p in out.rglob("*.json"))
    assert files == [
        "gemma-4-e2b__iphone-17-pro.json",
        "gemma-4-e2b__m4-max.json",
    ]
    assert runner.invoke(compat_app, ["device-run", "validate", str(out)]).exit_code == 0


def test_cli_ingest_compat_check_refusal_exits_1(tmp_path: Path) -> None:
    report = tmp_path / "compat.json"
    doc = load_json(LLM_SAMPLES / "compat_check" / "compat_0.15.0.json")
    doc["results"].append(
        {"repo": "x/broken", "file": "b.litertlm", "lane": "llm",
         "status": "BROKEN", "error": "boom", "answer": None}
    )
    report.write_text(canonical_dumps(doc), encoding="utf-8")
    out = tmp_path / "device_runs"
    result = runner.invoke(
        compat_app,
        [
            "device-run", "ingest-compat-check", str(report),
            "--date", "2026-08-10", "--device", "test-mac",
            "--accelerator", "cpu", "--out", str(out),
        ],
    )
    assert result.exit_code == 1  # the ok row ingested, the BROKEN row refused
    assert "refused" in result.stderr
    assert (out / "0.15.0" / "2026-08-10" / "granite-4.0-h-350m__test-mac.json").is_file()


# --- enrich --device-runs ----------------------------------------------------


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "data").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "cards", ws / "cards")
    shutil.copytree(DEVICE_RUN_EXAMPLES, ws / "data" / "examples" / "device_runs")
    (ws / "data" / "matrix").mkdir()
    return ws


def test_enrich_device_runs_idempotent_and_writes_no_matrix_bytes(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    shutil.copy(
        EXAMPLES / "matrix_npu_qnn_example.json",
        ws / "data" / "matrix" / "npu_qnn__2.1.6.json",
    )
    data_before = {
        str(p.relative_to(ws)): p.read_bytes()
        for p in sorted((ws / "data").rglob("*"))
        if p.is_file()
    }
    snapshot_dirs = [
        ws / "data" / "examples" / "device_runs" / "0.0.0-example" / "2026-01-01",
        ws / "data" / "examples" / "device_runs" / "0.0.0-example" / "2026-01-02",
    ]
    args = ["enrich"]
    for directory in snapshot_dirs:
        args += ["--device-runs", str(directory)]
    args.append(str(ws / "cards"))

    result = runner.invoke(cards_app, args)
    assert result.exit_code == 0, result.output
    first = {
        str(p.relative_to(ws)): p.read_bytes()
        for p in sorted(ws.rglob("*"))
        if p.is_file()
    }
    # The committed cards ARE this enrichment's output: byte-stable rerun.
    result = runner.invoke(cards_app, args)
    assert result.exit_code == 0, result.output
    second = {
        str(p.relative_to(ws)): p.read_bytes()
        for p in sorted(ws.rglob("*"))
        if p.is_file()
    }
    assert first == second
    # Zero matrix writes; in fact zero writes under data/ at all.
    data_after = {
        str(p.relative_to(ws)): p.read_bytes()
        for p in sorted((ws / "data").rglob("*"))
        if p.is_file()
    }
    assert data_after == data_before


def test_enrich_device_runs_conflicting_snapshots_refused(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    duplicate = ws / "data" / "examples" / "device_runs" / "0.0.0-example" / "2026-01-03"
    shutil.copytree(
        ws / "data" / "examples" / "device_runs" / "0.0.0-example" / "2026-01-01", duplicate
    )
    result = runner.invoke(
        cards_app,
        [
            "enrich",
            "--device-runs",
            str(ws / "data" / "examples" / "device_runs" / "0.0.0-example" / "2026-01-01"),
            "--device-runs", str(duplicate),
            str(ws / "cards"),
        ],
    )
    assert result.exit_code == 1
    assert "already loaded" in result.stderr


def test_load_device_runs_same_accelerator_new_prompt_length_coexists(tmp_path: Path) -> None:
    """A 205-token re-measurement of an accelerator that already has a
    19-token row is a new fact, not a duplicate (DECISIONS #162) — both
    snapshots load, so both rows reach the card."""
    root = tmp_path / "runs"
    write_snapshot(
        root, "0.0.0-example", "2026-01-01",
        [example_doc("m", "d", [example_record("gpu", metrics={"prefill_tokens": 19})])],
    )
    write_snapshot(
        root, "0.0.0-example", "2026-02-01",
        [example_doc("m", "d", [example_record("gpu", metrics={"prefill_tokens": 205})])],
    )
    loaded = load_device_runs(
        [root / "0.0.0-example" / "2026-01-01", root / "0.0.0-example" / "2026-02-01"]
    )
    assert len(loaded) == 2


def test_index_device_column_and_llms_txt() -> None:
    """The committed catalog outputs carry the device columns."""
    index = load_json(REPO_ROOT / "cards" / "index.json")
    assert index["schema_version"] == "1.2"
    by_id = {m["id"]: m for m in index["models"]}
    clean = by_id["example-tiny-clean"]["device"]
    assert [(r["device"], r["accelerator"], r["status"]) for r in clean["records"]] == [
        ("example-phone", "cpu_xnnpack", "pass"),
        ("example-phone", "npu_qnn", "fallback"),
    ]
    assert clean["records"][1]["runtime"] == "litert"
    mixed = by_id["example-tiny-mixed"]["device"]
    assert mixed["records"][0]["runtime"] == "litert-lm"
    assert by_id["example-web-a"]["device"] is None

    readme = (REPO_ROOT / "cards" / "README.md").read_text(encoding="utf-8")
    assert "| Device |" in readme
    assert "example-phone npu_qnn: fallback" in readme
    llms = (REPO_ROOT / "llms.txt").read_text(encoding="utf-8")
    assert "## Device-run results" in llms
    assert "device_runs/0.0.0-example/2026-01-01/example-tiny-clean__example-phone" in llms


# --- freshness delta, device lane --------------------------------------------


def _two_device_snapshots(root: Path) -> None:
    write_snapshot(
        root, "0.14.0", "2026-01-01",
        [
            example_doc(
                "model-a", "phone",
                [
                    example_record("gpu", runtime_version="0.14.0", decode=100.0),
                    example_record("npu", runtime_version="0.14.0", decode=None, latency=5.0),
                ],
            ),
            example_doc("model-gone", "phone", [example_record("gpu", runtime_version="0.14.0")]),
        ],
    )
    write_snapshot(
        root, "0.15.0", "2026-02-01",
        [
            example_doc(
                "model-a", "phone",
                [
                    example_record("gpu", runtime_version="0.15.0", decode=45.0),
                    example_record(
                        "npu", runtime_version="0.15.0", decode=None, latency=None,
                        loads=False, runs=False, failure_class="engine_create_failed",
                    ),
                ],
            ),
            example_doc("model-new", "phone", [example_record("gpu", runtime_version="0.15.0")]),
        ],
    )


def test_compute_device_delta_reports_all_change_kinds(tmp_path: Path) -> None:
    root = tmp_path / "device_runs"
    _two_device_snapshots(root)
    old, new = discover_device_run_snapshots(root)
    delta = compute_device_delta(old, new, metric_threshold_pct=25.0)
    assert delta["has_changes"]
    assert delta["runtime"] == "litert-lm"
    assert delta["regressions"] == [
        {
            "model_id": "model-a", "device": "phone", "accelerator": "npu",
            "old_status": "pass", "new_status": "load_failed",
            "failure_class": "engine_create_failed",
        }
    ]
    assert delta["metric_shifts"] == [
        {
            "model_id": "model-a", "device": "phone", "accelerator": "gpu",
            "metric": "decode_tokens_per_s", "old": 100.0, "new": 45.0,
            "change_pct": -55.0,
        }
    ]
    assert delta["models_added"] == ["model-new on phone"]
    assert delta["models_removed"] == ["model-gone on phone"]
    entry = render_device_entry(delta)
    assert device_entry_heading(delta) == (
        "## litert-lm 0.14.0 → 0.15.0 (device runs 2026-01-01 → 2026-02-01)"
    )
    assert "decode_tokens_per_s: 100 → 45 (-55%)" in entry


def test_compute_device_delta_condition_change_is_not_a_metric_shift(tmp_path: Path) -> None:
    """A prompt-length change between snapshots (19tok floor -> 205tok) must
    read as record added/removed, never as a throughput shift — the two
    conditions are different measurements (DECISIONS #162)."""
    root = tmp_path / "device_runs"
    write_snapshot(
        root, "0.14.0", "2026-01-01",
        [example_doc("model-a", "phone", [example_record(
            "gpu", runtime_version="0.14.0", decode=100.0, metrics={"prefill_tokens": 19},
        )])],
    )
    write_snapshot(
        root, "0.15.0", "2026-02-01",
        [example_doc("model-a", "phone", [example_record(
            "gpu", runtime_version="0.15.0", decode=35.0, metrics={"prefill_tokens": 205},
        )])],
    )
    old, new = discover_device_run_snapshots(root)
    delta = compute_device_delta(old, new, metric_threshold_pct=25.0)
    assert delta["metric_shifts"] == []
    assert delta["regressions"] == []
    assert delta["records_added"] == ["model-a on phone x gpu@205tok"]
    assert delta["records_removed"] == ["model-a on phone x gpu@19tok"]


def test_device_delta_upsert_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "device_runs"
    _two_device_snapshots(root)
    old, new = discover_device_run_snapshots(root)
    delta = compute_device_delta(old, new, metric_threshold_pct=25.0)
    log = tmp_path / "DELTA_LOG.md"
    assert upsert_entry(log, delta)
    first = log.read_bytes()
    assert not upsert_entry(log, delta)
    assert log.read_bytes() == first


def test_delta_cli_covers_device_snapshots(tmp_path: Path) -> None:
    root = tmp_path / "device_runs"
    _two_device_snapshots(root)
    log = tmp_path / "DELTA_LOG.md"
    args = [
        "freshness", "delta",
        "--sweep-dir", str(tmp_path / "no-sweep"),
        "--device-runs-dir", str(root),
        "--log", str(log), "--json",
    ]
    result = runner.invoke(compat_app, args)
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["has_changes"]
    assert len(payload["device_deltas"]) == 1
    assert payload["device_deltas"][0]["runtime"] == "litert-lm"
    content = log.read_text(encoding="utf-8")
    assert "## litert-lm 0.14.0 → 0.15.0 (device runs" in content

    rerun = runner.invoke(compat_app, args)
    assert rerun.stdout == result.stdout  # deterministic
    assert log.read_text(encoding="utf-8") == content  # idempotent upsert


def test_delta_cli_diffs_per_runtime_never_across(tmp_path: Path) -> None:
    """litert and litert-lm snapshots interleave; only same-runtime pairs diff.
    One snapshot per runtime -> nothing to diff, exit 0, no log written."""
    root = tmp_path / "device_runs"
    write_snapshot(
        root, "0.15.0", "2026-01-01",
        [example_doc("m", "d", [example_record("gpu", runtime_version="0.15.0")])],
    )
    write_snapshot(
        root, "2.1.6", "2026-01-02",
        [
            example_doc(
                "m", "d",
                [example_record("npu_qnn", runtime="litert", runtime_version="2.1.6")],
            )
        ],
    )
    result = runner.invoke(
        compat_app,
        [
            "freshness", "delta",
            "--sweep-dir", str(tmp_path / "no-sweep"),
            "--device-runs-dir", str(root),
            "--log", str(tmp_path / "x.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "x.md").exists()


# --- lane 1b: litert_lm_main / litert-lm benchmark console logs -------------
#
# Built against the real staged excerpts in data/examples/llm_samples/console_log/
# (Pixel 8a, litert-lm 0.16.0 tag build, 2026-08-12): one fully delegated PASS
# and one engine-abort FAIL whose rejection list names the ops.

CONSOLE_SAMPLES = LLM_SAMPLES / "console_log"


def test_console_log_pass_sample_parses_speeds_and_delegation() -> None:
    from litert_compat.device_runs.adapters import parse_console_log

    log = parse_console_log(
        (CONSOLE_SAMPLES / "lfm25-12b-int4-gpu.pixel8a.cl.pass.log").read_text()
    )
    # Largest subgraph, not the sum: prefill and decode are separate signatures
    # of one model and adding them would invent a node count.
    assert (log.delegated_ops, log.total_ops) == (542, 542)
    assert log.prefill_tok_s == 67.46
    assert log.decode_tok_s == 19.88
    assert log.ttft_s == 0.33
    assert log.init_ms == 9787.20
    assert log.prefill_tokens == 19 and log.decode_tokens == 50
    assert log.has_results and not log.engine_failed
    assert log.unsupported_ops == []


def test_console_log_fail_sample_names_ops_and_marks_abort() -> None:
    from litert_compat.device_runs.adapters import parse_console_log

    log = parse_console_log(
        (CONSOLE_SAMPLES / "granite-4.0-h-350m.pixel8a.cl.fail.log").read_text()
    )
    assert (log.delegated_ops, log.total_ops) == (109, 3673)
    assert log.engine_failed and not log.has_results
    for op in ("MUL", "EXP", "CUMSUM", "BROADCAST_TO", "CAST"):
        assert op in log.unsupported_ops
    assert any("must be less than 5" in line for line in log.unsupported_lines)


def test_console_record_verdicts_follow_the_log_body() -> None:
    from litert_compat.device_runs.adapters import (
        console_record,
        make_env,
        parse_console_log,
    )

    env = make_env(device="Pixel 8a", runtime="litert-lm", runtime_version="0.16.0")
    ok = console_record(
        parse_console_log(
            (CONSOLE_SAMPLES / "lfm25-12b-int4-gpu.pixel8a.cl.pass.log").read_text()
        ),
        accelerator="gpu",
        env=env,
        date="2026-08-12",
    )
    assert ok["loads"] and ok["runs"] and ok["failure_class"] is None
    assert ok["full_delegation"] is True
    assert ok["decode_tokens_per_s"] == 19.88
    assert ok["ttft_ms"] == 330.0

    bad = console_record(
        parse_console_log(
            (CONSOLE_SAMPLES / "granite-4.0-h-350m.pixel8a.cl.fail.log").read_text()
        ),
        accelerator="gpu",
        env=env,
        date="2026-08-12",
    )
    assert not bad["runs"] and bad["failure_class"] == "engine_create_failed"
    assert bad["full_delegation"] is False
    assert bad["decode_tokens_per_s"] is None


def test_console_record_takes_token_counts_the_log_lost_to_tail() -> None:
    """A `tail -12` capture keeps the results block and cuts the token-count
    headers above it, so the record would carry no prompt-length condition and
    read as a different fact from every other 256-token cell. The counts the
    run's own traced command line states are supplied by the caller instead."""
    from litert_compat.device_runs.adapters import console_record, make_env, parse_console_log
    from litert_compat.device_runs.records import record_label

    tail_cut = (
        "+ litert-lm benchmark model.litertlm -p 256 -d 256 --runs 3 --backend cpu\n"
        "+ tail -12\n"
        "Max number of tokens       : 4096\n"
        "----- Results -----\n"
        "Prefill speed:        97.25 tokens/s\n"
        "Decode speed:         20.18 tokens/s\n"
        "Time to first token:  2.6820 s\n"
    )
    log = parse_console_log(tail_cut)
    assert log.prefill_tokens is None and log.decode_tokens is None

    bare = console_record(
        log,
        accelerator="cpu",
        env=make_env(device="Mac Studio M4 Max", runtime="litert-lm", runtime_version="0.16.0"),
        date="2026-08-25",
    )
    assert "metrics" not in bare
    assert record_label(bare) == "cpu"

    supplied = console_record(
        log,
        accelerator="cpu",
        env=make_env(device="Mac Studio M4 Max", runtime="litert-lm", runtime_version="0.16.0"),
        date="2026-08-25",
        prefill_tokens=256,
        decode_tokens=256,
    )
    assert supplied["metrics"] == {"prefill_tokens": 256.0, "decode_tokens": 256.0}
    assert record_label(supplied) == "cpu@256tok"
    assert any("supplied by the caller" in line for line in supplied["evidence"])


def test_console_record_refuses_token_counts_that_contradict_the_log() -> None:
    """The log's own header outranks the command line: a supplied count that
    disagrees with one the log states means the count or the slice is wrong."""
    from litert_compat.device_runs.adapters import (
        AdapterError,
        console_record,
        make_env,
        parse_console_log,
    )

    log = parse_console_log(
        "Number of tokens in prefill: 256\n"
        "Number of tokens in decode : 256\n"
        "Prefill speed:        210.95 tokens/s\n"
        "Decode speed:         33.97 tokens/s\n"
    )
    env = make_env(device="Mac Studio M4 Max", runtime="litert-lm", runtime_version="0.16.0")
    with pytest.raises(AdapterError, match="contradicts the log"):
        console_record(log, accelerator="cpu", env=env, date="2026-08-27", prefill_tokens=205)

    # Agreeing with the log is allowed and adds no caller-supplied evidence:
    # the log stated the condition itself.
    agreed = console_record(
        log, accelerator="cpu", env=env, date="2026-08-27", prefill_tokens=256, decode_tokens=256
    )
    assert agreed["metrics"] == {"prefill_tokens": 256.0, "decode_tokens": 256.0}
    assert not any("supplied by the caller" in line for line in agreed["evidence"])


def test_console_log_rejects_a_log_it_cannot_read() -> None:
    from litert_compat.device_runs.adapters import AdapterError, parse_console_log

    with pytest.raises(AdapterError):
        parse_console_log("")
    with pytest.raises(AdapterError):
        parse_console_log("some unrelated program output\nwith no runtime lines\n")


def test_console_log_counts_only_the_named_delegate() -> None:
    """A partially delegated graph also prints XNNPACK lines for the CPU
    remainder; counting both would report more nodes than the graph has."""
    from litert_compat.device_runs.adapters import parse_console_log

    text = (
        "VERBOSE: Replacing 48 out of 90 node(s) with delegate (LITERT_CL) node, "
        "yielding 2 partitions for subgraph 0 (main).\n"
        "VERBOSE: Replacing 32 out of 43 node(s) with delegate "
        "(TfLiteXNNPackDelegate) node, yielding 4 partitions for subgraph 0 (main).\n"
    )
    log = parse_console_log(text)
    assert (log.delegated_ops, log.total_ops) == (48, 90)
    xnn = parse_console_log(text, delegate_tag="TfLiteXNNPackDelegate")
    assert (xnn.delegated_ops, xnn.total_ops) == (32, 43)


# --- lane 5: iOS BenchmarkApp yardstick JSON ---------------------------------


def _yardstick_runs():
    from litert_compat.device_runs.adapters import parse_yardstick

    files = sorted((LLM_SAMPLES / "yardstick").glob("*.json"))
    assert len(files) == 2
    return [parse_yardstick(load_json(f)) for f in files]


def test_yardstick_parses_the_staged_samples() -> None:
    runs = _yardstick_runs()
    by_task = {r.task: r for r in runs}
    q = by_task["quality"]
    assert q.generated_tokens == 39 and q.prompt_tokens == 132
    assert q.decode_tok_s is not None and 13.6 < q.decode_tok_s < 13.7
    assert q.peak_mem_mb is not None and q.device_model == "iPhone18,1"
    assert by_task["vision-no-text"].output_sample == "No"


def test_yardstick_merges_tasks_into_one_record_honestly() -> None:
    from litert_compat.device_runs.adapters import make_env, yardstick_date, yardstick_record
    from litert_compat.device_runs.records import device_run_errors

    runs = _yardstick_runs()
    assert yardstick_date(runs) == "2026-08-19"  # the source's own UTC stamp
    rec = yardstick_record(
        runs, accelerator="gpu",
        env=make_env(device="iPhone 17 Pro", runtime="litert-lm", runtime_version="0.16.0"),
        date="2026-08-19",
    )
    # throughput comes from the longest generation (quality, 39 tokens), not the 2-token probe
    assert rec["decode_tokens_per_s"] == round(runs[[r.task for r in runs].index("quality")].decode_tok_s, 2)
    assert rec["ttft_ms"] == 685.0
    # peak memory = max over tasks (the vision probe resident set is the larger one)
    assert rec["peak_mem_mb"] > 2900
    assert rec["output_match"] is None  # expected answers are not in the file
    assert any("output='No'" in e for e in rec["evidence"])
    assert "quality.generated_tokens" in rec["metrics"]
    doc = {"schema_version": "1.0", "model_id": "x", "device": "iphone-17-pro",
           "artifact": "x.litertlm", "quantization": None, "results": [rec]}
    assert device_run_errors(doc) == []


def test_yardstick_refuses_other_runtimes_and_malformed() -> None:
    from litert_compat.device_runs.adapters import AdapterError, parse_yardstick

    doc = load_json(next((LLM_SAMPLES / "yardstick").glob("*.json")))
    with pytest.raises(AdapterError):
        parse_yardstick({**doc, "runtime": "core-ai"})
    with pytest.raises(AdapterError):
        parse_yardstick({"task": "quality"})


def test_yardstick_aggregates_bench_protocol_runs() -> None:
    """N runs of ONE task (cold-warm-split): headline = warm median, cold and
    spread kept in metrics, every raw run in the evidence."""
    from litert_compat.device_runs.adapters import make_env, parse_yardstick, yardstick_record
    from litert_compat.device_runs.records import device_run_errors

    template = load_json(next((LLM_SAMPLES / "yardstick").glob("*.quality.json")))
    docs = []
    for i, (cold, decode) in enumerate([(True, 90.0), (False, 100.0), (False, 110.0), (False, 120.0)]):
        doc = json.loads(json.dumps(template))
        doc["task"] = "short-chat"
        doc["timestamp"] = f"2026-08-27T00:0{i}:00Z"
        doc["metrics"]["coldRun"] = cold
        doc["metrics"]["decodeTokensPerSecond"] = decode
        doc["metrics"]["contextTokensConfigured"] = 1024
        docs.append(doc)
    runs = [parse_yardstick(d) for d in docs]
    rec = yardstick_record(
        runs, accelerator="gpu",
        env=make_env(device="iPhone 17 Pro", runtime="litert-lm", runtime_version="0.16.0"),
        date="2026-08-27",
    )
    assert rec["decode_tokens_per_s"] == 110.0          # median of the 3 warm runs
    assert rec["context_length"] == 1024                 # stamped per run, unanimous
    m = rec["metrics"]
    assert m["short_chat.n_runs"] == 4.0
    assert m["short_chat.cold_decode_tokens_per_s"] == 90.0
    assert m["short_chat.decode_spread_pct"] == round((120.0 - 100.0) / 110.0 * 100, 2)
    assert any("median of 3 warm run(s)" in e for e in rec["evidence"])
    assert sum(1 for e in rec["evidence"] if e.startswith("[short-chat ")) == 4  # every raw run kept
    doc = {"schema_version": "1.0", "model_id": "x", "device": "iphone-17-pro",
           "artifact": "x.litertlm", "quantization": None, "results": [rec]}
    assert device_run_errors(doc) == []


# --- lane 6: npubench classic sweep ------------------------------------------

NPUBENCH_SAMPLE = (LLM_SAMPLES / "npubench" / "results_sample.jsonl").read_text()


def _npubench(text: str = NPUBENCH_SAMPLE):
    from litert_compat.device_runs.adapters import npubench_records

    return npubench_records(
        text,
        runtime_version="2.2.0",
        device="galaxy-s26",
        device_name="Galaxy S26 (SM-S942Q)",
        soc="Qualcomm SM8850",
        os_build="Android 16",
        machine_label="galaxy-s26-npubench",
    )


def test_npubench_accepted_jit_pair_maps_median_and_cold_load() -> None:
    results, _ = _npubench()
    by_id = {r.model_id: r for r in results}
    cpga = by_id["cpga-net-lowlight"]
    npu = next(r for r in cpga.records if r["accelerator"] == "npu_qnn")
    gpu = next(r for r in cpga.records if r["accelerator"] == "gpu_mldrift")
    assert npu["loads"] and npu["runs"] and npu["failure_class"] is None
    assert npu["latency_p50_ms"] == 4.62
    # the cold row's load includes the on-device JIT compile
    assert npu["metrics"]["jit_first_load_ms"] == 2238.0
    assert npu["env"]["vendor_sdk"] == "QAIRT (Hexagon, JIT on-device)"
    assert gpu["latency_p50_ms"] == 14.94
    assert gpu["env"]["vendor_sdk"] is None
    # prefill/decode stay null: classic models have no token throughput
    assert npu["prefill_tokens_per_s"] is None


def test_npubench_aot_mode_lands_in_vendor_sdk() -> None:
    results, _ = _npubench()
    aot = [
        r for res in results for r in res.records
        if r["accelerator"] == "npu_qnn" and r["failure_class"] is None
        and "AOT" in (r["env"]["vendor_sdk"] or "")
    ]
    assert aot, "the staged sample carries one accepted AOT row"
    assert "jit_first_load_ms" not in (aot[0].get("metrics") or {})


def test_npubench_failure_classes_are_the_staged_vocabulary() -> None:
    results, _ = _npubench()
    classes = {
        r["failure_class"]
        for res in results for r in res.records
        if r["failure_class"] is not None
    }
    assert classes == {
        "process_crashed", "compile_failed", "load_failed",
        "invoke_failed", "aot_compile_failed", "silent_cpu_fallback",
    }
    # a silent fallback ran, but its number is never a measurement
    fallback = [
        r for res in results for r in res.records
        if r["failure_class"] == "silent_cpu_fallback"
    ]
    assert fallback[0]["latency_p50_ms"] is None


def test_npubench_skips_soc_mismatch_and_unobtainable() -> None:
    results, skipped = _npubench()
    ids = {r.model_id for r in results}
    assert not any("sam2" in i for i in ids)
    assert any("soc-mismatch" in s for s in skipped)
    assert any("TimeoutExpired" in s for s in skipped)


def test_npubench_annul_voids_prior_rows() -> None:
    import json as _json

    rows = [_json.loads(line) for line in NPUBENCH_SAMPLE.splitlines() if line.strip()]
    accepted = next(
        r for r in rows
        if r.get("accepted") and r.get("backend") == "npu" and r.get("phase") == "cached"
    )
    annul = {
        "annul": True, "repo": accepted["repo"], "slug": accepted["slug"],
        "backend": "npu", "reason": "test", "ts": "2026-08-27T00:00:00",
    }
    text = "\n".join(
        _json.dumps(r) for r in rows + [annul]
    )
    results, _ = _npubench(text)
    for res in results:
        if res.artifact == accepted["file"].rsplit("/", 1)[-1]:
            assert all(r["accelerator"] != "npu_qnn" for r in res.records)


def test_npubench_unknown_error_shape_is_refused_not_guessed() -> None:
    import json as _json

    row = {
        "repo": "litert-community/X-LiteRT", "slug": "X-LiteRT__x", "file": "x.tflite",
        "backend": "npu", "phase": "cold", "mode": "jit", "ok": False,
        "accepted": False, "error": "SomethingNeverStaged: boom",
        "ts": "2026-08-27T00:00:00",
    }
    results, skipped = _npubench(_json.dumps(row))
    assert results == []
    assert any("refusing to classify" in s for s in skipped)


# --- 2026-09-02: artifact digest, signature dimension, three journal shapes ---


def test_schema_accepts_artifact_sha256_and_signature_and_rejects_bad_digest() -> None:
    doc = example_doc("m", "dev", [example_record()])
    doc["artifact_sha256"] = "a" * 64
    doc["results"][0]["signature"] = "decode"
    assert device_run_errors(doc) == []
    doc["artifact_sha256"] = "not-a-digest"
    assert device_run_errors(doc)


def test_record_label_carries_signature_but_condition_wins() -> None:
    rec = example_record()
    rec["metrics"] = {}
    assert record_label(rec) == rec["accelerator"]
    rec["signature"] = "prefill_128"
    assert record_label(rec) == f"{rec['accelerator']}@prefill_128"
    rec["metrics"] = {"prefill_tokens": 256}
    assert record_label(rec) == f"{rec['accelerator']}@256tok"


def test_write_refuses_two_digests_under_one_artifact_name(tmp_path: Path) -> None:
    doc = example_doc("m", "dev", [example_record()])
    doc["artifact_sha256"] = "1" * 64
    date = doc["results"][0]["date"]
    write_device_run(tmp_path, doc, date=date)
    other = example_doc("m", "dev", [example_record()])
    other["artifact_sha256"] = "2" * 64
    other["results"][0]["signature"] = "other"
    with pytest.raises(DeviceRunError, match="artifact_sha256"):
        write_device_run(tmp_path, other, date=date)
    # a digest arriving for a file that had none is carried into the merge
    third = example_doc("m", "dev", [example_record()])
    third["results"][0]["signature"] = "third"
    target = write_device_run(tmp_path, third, date=date)
    assert load_json(target)["artifact_sha256"] == "1" * 64


P8A_SAMPLE = (LLM_SAMPLES / "npubench" / "results_p8a_sample.jsonl").read_text()
P8A_REPOS = {
    "sam2_tiny_image_encoder_v2_fp16.tflite": "litert-community/SAM2.1-Hiera-Tiny-Image-Encoder",
    "sam2_tiny_image_encoder_fp16.tflite": "litert-community/SAM2.1-Hiera-Tiny-Image-Encoder",
}


def test_npubench_repo_for_fills_missing_repo_and_cpu_rows_land_as_cpu_xnnpack() -> None:
    from litert_compat.device_runs.adapters import npubench_records

    with pytest.raises(AdapterError, match="no repo/slug"):
        npubench_records(P8A_SAMPLE, runtime_version="2.2.0", device="pixel-8a", device_name="Pixel 8a")
    results, skipped = npubench_records(
        P8A_SAMPLE, runtime_version="2.2.0", device="pixel-8a", device_name="Pixel 8a",
        repo_for=P8A_REPOS,
    )
    by_id = {r.model_id: r for r in results}
    v2 = by_id["sam2.1-hiera-tiny-image-encoder__sam2_tiny_image_encoder_v2_fp16"]
    accels = {r["accelerator"]: r for r in v2.records}
    assert set(accels) == {"gpu_mldrift", "cpu_xnnpack"}
    assert accels["gpu_mldrift"]["latency_p50_ms"] == 576.689
    assert accels["gpu_mldrift"]["full_delegation"] is True
    assert accels["gpu_mldrift"]["delegated_ops"] == 867
    assert accels["cpu_xnnpack"]["latency_p50_ms"] == 10545.15
    assert accels["cpu_xnnpack"]["full_delegation"] is False  # 866/867 on XNNPACK
    # the unaccepted (thermal LIGHT) attempt of the other file produces no record
    assert "sam2.1-hiera-tiny-image-encoder__sam2_tiny_image_encoder_fp16" not in by_id


def test_npubench_repo_for_refuses_a_contradicting_repo() -> None:
    from litert_compat.device_runs.adapters import npubench_records

    row = json.loads(P8A_SAMPLE.splitlines()[0])
    row["repo"] = "litert-community/Somewhere-Else"
    with pytest.raises(AdapterError, match="maps"):
        npubench_records(
            json.dumps(row), runtime_version="2.2.0", device="pixel-8a", device_name="Pixel 8a",
            repo_for=P8A_REPOS,
        )


PI5_SAMPLE = (LLM_SAMPLES / "pi5" / "results_sample.jsonl").read_text()


def test_pi5_rows_map_median_footprint_version_and_signature() -> None:
    from litert_compat.device_runs.adapters import pi5_benchmark_records

    results, skipped = pi5_benchmark_records(PI5_SAMPLE, device="raspberry-pi-5")
    assert skipped == []
    by_id = {r.model_id: r for r in results}
    mlsd = by_id["m-lsd-tiny"]
    rec = mlsd.records[0]
    assert rec["accelerator"] == "cpu_xnnpack" and rec["loads"] and rec["runs"]
    assert rec["latency_p50_ms"] == 106.619
    assert rec["peak_mem_mb"] == 74.56
    assert rec["metrics"]["iterations"] == 150 and rec["metrics"]["threads"] == 4
    assert rec["env"]["runtime"] == "litert"
    assert rec["env"]["runtime_version"] == "2.2.0.dev20260804"  # read from the row
    assert rec["env"]["device"] == "Raspberry Pi 5 Model B Rev 1.1"
    assert rec["date"] == "2026-08-31" and "signature" not in rec
    talker = by_id["qwen3-tts-12hz-0.6b-base"]
    assert talker.records[0]["signature"] == "decode"
    assert record_label(talker.records[0]) == "cpu_xnnpack@decode"


def test_pi5_unaccepted_or_errored_rows_are_skipped_not_guessed() -> None:
    from litert_compat.device_runs.adapters import pi5_benchmark_records

    row = json.loads(PI5_SAMPLE.splitlines()[0])
    row["invocations"][1]["throttled_after"] = "0x50000"
    results, skipped = pi5_benchmark_records(json.dumps(row), device="raspberry-pi-5")
    assert results == [] and any("not accepted" in s for s in skipped)
    row = json.loads(PI5_SAMPLE.splitlines()[0])
    row["error"] = "download failed"
    results, skipped = pi5_benchmark_records(json.dumps(row), device="raspberry-pi-5")
    assert results == [] and any("refusing to classify" in s for s in skipped)


PARITY_SAMPLE = json.loads((LLM_SAMPLES / "npubench" / "parity_report_sample.json").read_text())


def test_parity_report_maps_ran_and_failed_legs_per_signature() -> None:
    from litert_compat.device_runs.adapters import npubench_parity_records

    results, skipped = npubench_parity_records(
        PARITY_SAMPLE, repo="litert-community/granite-speech-5.0-470m-turboctc",
        files={"wi8fc": "granite_speech_ctc_wi8fc.tflite", "fp16": "granite_speech_ctc_fp16.tflite"},
        signature_for={"10s": "transcribe_10s"}, runtime_version="2.2.0", date="2026-09-01",
        device="galaxy-s26", device_name="Galaxy S26 (SM-S942Q)",
    )
    assert skipped == []
    (res,) = results
    assert res.model_id == "granite-speech-5.0-470m-turboctc__granite_speech_ctc_wi8fc"
    cpu = next(r for r in res.records if r["accelerator"] == "cpu")
    gpu = next(r for r in res.records if r["accelerator"] == "gpu_mldrift")
    assert cpu["signature"] == "transcribe_10s" and cpu["latency_p50_ms"] == 220.226
    assert cpu["metrics"]["logit_maxdiff_mac"] == 0.0 and cpu["output_match"] is None
    assert gpu["failure_class"] == "compile_failed" and gpu["loads"] is False
    assert record_label(cpu) == "cpu@transcribe_10s"
    with pytest.raises(AdapterError, match="no --file mapping"):
        npubench_parity_records(
            PARITY_SAMPLE, repo="x/y", files={}, runtime_version="2.2.0", date="2026-09-01",
            device="d", device_name="D",
        )


PI5_LLM_SAMPLE = (LLM_SAMPLES / "pi5" / "llm_results_sample.jsonl").read_text()


def test_pi5_llm_rows_map_medians_rss_version_gate_and_ids() -> None:
    from litert_compat.device_runs.adapters import pi5_llm_records

    results, skipped = pi5_llm_records(PI5_LLM_SAMPLE, device="raspberry-pi-5")
    assert skipped == []
    by_id = {r.model_id: r for r in results}
    # multi-bundle repo -> repo id + artifact suffix (the compat_check convention)
    lfm = by_id["lfm2.5-1.2b-instruct-int4"]
    assert lfm.artifact == "LFM2.5-1.2B-Instruct_int4.litertlm" and lfm.date == "2026-09-01"
    rec = lfm.records[0]
    assert rec["accelerator"] == "cpu" and rec["loads"] and rec["runs"]
    assert rec["prefill_tokens_per_s"] == 54.09 and rec["decode_tokens_per_s"] == 9.29  # medians
    assert rec["ttft_ms"] == 4841.0 and rec["peak_mem_mb"] == 1455
    m = rec["metrics"]
    assert m["prefill_tokens"] == 256 and m["decode_tokens"] == 256 and m["threads"] == 4
    assert m["init_s"] == 33.8695 and m["decode_tps_min"] == 9.26 and m["decode_tps_max"] == 9.34
    assert rec["env"]["runtime"] == "litert-lm"
    assert rec["env"]["runtime_version"] == "0.16.1"  # read from the row's versions
    assert rec["env"]["device"] == "Raspberry Pi 5 Model B Rev 1.1"
    assert rec["env"]["os_build"].startswith("Linux-6.18")
    assert any("gate (" in e and "status pass" in e for e in rec["evidence"])
    assert "lfm2.5-1.2b-instruct-int8" in by_id
    # single-bundle repo keeps the bare repo id
    assert by_id["granite-4.0-h-1b"].artifact == "granite-4.0-h-1b_int8.litertlm"
    # caller-supplied catalog ids win over the mechanical id (repo-qualified or bare file key)
    results, _ = pi5_llm_records(
        PI5_LLM_SAMPLE, device="raspberry-pi-5",
        model_id_for={"LFM2.5-1.2B-Instruct/LFM2.5-1.2B-Instruct_int4.litertlm": "lfm-int4-card",
                      "granite-4.0-h-1b_int8.litertlm": "granite-card"},
    )
    ids = {r.model_id for r in results}
    assert {"lfm-int4-card", "granite-card", "lfm2.5-1.2b-instruct-int8"} == ids


def test_pi5_llm_unaccepted_rows_are_skipped_not_guessed() -> None:
    from litert_compat.device_runs.adapters import pi5_llm_records

    first = PI5_LLM_SAMPLE.splitlines()[0]
    row = json.loads(first)
    row["gate"]["status"] = "degenerate"
    results, skipped = pi5_llm_records(json.dumps(row), device="raspberry-pi-5")
    assert results == [] and any("gate not passed" in s for s in skipped)
    row = json.loads(first)
    row["invocations"][2]["throttled_after"] = "0x50000"
    results, skipped = pi5_llm_records(json.dumps(row), device="raspberry-pi-5")
    assert results == [] and any("not accepted" in s for s in skipped)
    row = json.loads(first)
    row["backend"] = "gpu"
    results, skipped = pi5_llm_records(json.dumps(row), device="raspberry-pi-5")
    assert results == [] and any("no staged sample" in s for s in skipped)
    row = json.loads(first)
    row["error"] = "download failed"
    results, skipped = pi5_llm_records(json.dumps(row), device="raspberry-pi-5")
    assert results == [] and any("refusing to classify" in s for s in skipped)
    # the same (repo, file) twice: first occurrence kept, the repeat noted
    results, skipped = pi5_llm_records(first + "\n" + first, device="raspberry-pi-5")
    assert len(results) == 1 and any("duplicate row" in s for s in skipped)


def test_compat_check_model_id_for_overrides_the_mechanical_id() -> None:
    report = {
        "runtime": "~/x/bin/litert-lm", "runtime_version": "0.17.0", "checked": 2, "broken": 0,
        "results": [
            {"repo": "litert-community/LFM2.5-1.2B-Thinking", "file": "LFM2.5-1.2B-Thinking_int4.litertlm",
             "lane": "llm", "status": "ok", "answer": "42"},
            {"repo": "litert-community/Foo", "file": "model.litertlm", "lane": "llm", "status": "ok", "answer": "42"},
        ],
    }
    _, results, refused = compat_check_records(report, date="2026-09-06", device="mac", accelerator="cpu")
    assert refused == [] and {r.model_id for r in results} == {"lfm2.5-1.2b-thinking", "foo"}
    _, results, _ = compat_check_records(
        report, date="2026-09-06", device="mac", accelerator="cpu",
        model_id_for={"LFM2.5-1.2B-Thinking/LFM2.5-1.2B-Thinking_int4.litertlm": "lfm2.5-1.2b-thinking-int4"},
    )
    by = {r.artifact: r.model_id for r in results}
    assert by["LFM2.5-1.2B-Thinking_int4.litertlm"] == "lfm2.5-1.2b-thinking-int4" and by["model.litertlm"] == "foo"
