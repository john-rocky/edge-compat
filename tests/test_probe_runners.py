"""Backend runners: real cpu measurement, seeded-input determinism, graceful
unavailability, adb protocol parsing, and skip-marked hardware tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from litert_compat.probe.gen import ProbeFixture, generate_fixture
from litert_compat.probe.runners import (
    DEFAULT_ADB_HELPER,
    DEFAULT_NPU_ADB_HELPER,
    MATRIX_STATUSES,
    AdbMlDriftRunner,
    CpuRunner,
    NpuAdbRunner,
    RunnerError,
    WebGpuMacRunner,
    WebGpuPlaywrightRunner,
    _mulberry32,
    runner_registry,
)
from litert_compat.probe.signatures import make_signature

CPU_UNAVAILABLE = CpuRunner().availability()
requires_cpu = pytest.mark.skipif(
    CPU_UNAVAILABLE is not None,
    reason=f"cpu runner unavailable: {CPU_UNAVAILABLE}",
)


def fixture_for(
    op: str,
    tmp_path: Path,
    dtypes: tuple[str, ...] = ("float32",),
    meta: dict | None = None,
    backend: str = "cpu_xnnpack",
) -> ProbeFixture:
    sig = make_signature(
        backend, op, list(dtypes), meta or {"dynamic_shape": False, "rank": 4}
    )
    return generate_fixture(sig, tmp_path)


def test_mulberry32_deterministic_and_in_range() -> None:
    a = _mulberry32(12345)
    b = _mulberry32(12345)
    values = [a() for _ in range(100)]
    assert values == [b() for _ in range(100)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert _mulberry32(1)() != _mulberry32(2)()


def test_registry_names_and_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITERT_COMPAT_ADB_HELPER", raising=False)
    monkeypatch.delenv("LITERT_COMPAT_NPU_ADB_HELPER", raising=False)
    registry = runner_registry()
    assert sorted(registry) == [
        "cpu",
        "gpu_mldrift_adb",
        "npu_adb",
        "webgpu_mac",
        "webgpu_playwright",
    ]
    assert registry["cpu"].claims("cpu_xnnpack")
    assert not registry["cpu"].claims("gpu_mldrift")
    assert registry["gpu_mldrift_adb"].claims("gpu_mldrift")
    assert registry["webgpu_mac"].claims("webgpu_mac")
    # Phase 13: npu_adb claims exactly the registered NPU backend IDs.
    assert registry["npu_adb"].claims("npu_qnn")
    assert registry["npu_adb"].claims("npu_neuropilot")
    assert not registry["npu_adb"].claims("gpu_mldrift")
    assert not registry["gpu_mldrift_adb"].claims("npu_qnn")
    # Distinct default helpers: the NPU helper is a separate owner-side build.
    assert registry["gpu_mldrift_adb"].helper == DEFAULT_ADB_HELPER  # type: ignore[attr-defined]
    assert registry["npu_adb"].helper == DEFAULT_NPU_ADB_HELPER  # type: ignore[attr-defined]


@requires_cpu
def test_seeded_feeds_deterministic(tmp_path: Path) -> None:
    from litert_compat.probe.runners import seeded_feeds

    fixture = fixture_for("ADD", tmp_path, meta={"dynamic_shape": False, "rank": 3})
    first = seeded_feeds(fixture)
    second = seeded_feeds(fixture)
    assert len(first) == 2
    for (_, a), (_, b) in zip(first, second, strict=True):
        assert (a == b).all()
    # Positive domain keeps LOG/SQRT/DIV probes NaN-free.
    assert all(float(array.min()) >= 0.25 for _, array in first)


@requires_cpu
@pytest.mark.parametrize(
    ("op", "meta"),
    [
        ("ABS", {"dynamic_shape": False, "rank": 4}),
        ("SOFTMAX", {"dynamic_shape": False, "rank": 2}),
        ("RESHAPE", {"dynamic_shape": False, "rank": 2}),
        ("ADD", {"dynamic_shape": False, "rank": 3}),
        ("LOG", {"dynamic_shape": False, "rank": 1}),
        ("NEG", {"dynamic_shape": True, "rank": 2}),
        # Weighted/structured templates (the graduated deferred item) execute
        # for real too — constant weights, options, and honest op versions.
        ("CONV_2D", {"dynamic_shape": False, "rank": 4}),
        ("DEPTHWISE_CONV_2D", {"dynamic_shape": False, "rank": 4}),
        ("TRANSPOSE_CONV", {"dynamic_shape": False, "rank": 4}),
        ("FULLY_CONNECTED", {"dynamic_shape": False, "rank": 2}),
        ("FULLY_CONNECTED", {"dynamic_shape": False, "rank": 3}),
        ("BATCH_MATMUL", {"dynamic_shape": False, "rank": 3}),
        ("MAX_POOL_2D", {"dynamic_shape": False, "rank": 4}),
        ("SUM", {"dynamic_shape": False, "rank": 2}),
        ("TRANSPOSE", {"dynamic_shape": False, "rank": 5}),
        ("SLICE", {"dynamic_shape": False, "rank": 3}),
        ("PAD", {"dynamic_shape": False, "rank": 4}),
        ("CONCATENATION", {"dynamic_shape": False, "rank": 2}),
        ("LEAKY_RELU", {"dynamic_shape": False, "rank": 3}),
        ("RESIZE_BILINEAR", {"dynamic_shape": False, "rank": 4}),
        ("EMBEDDING_LOOKUP", {"dynamic_shape": False, "rank": 2}),
    ],
)
def test_cpu_runner_measures_for_real(op: str, meta: dict, tmp_path: Path) -> None:
    """DoD: the cpu runner measures the example fixtures for real — every
    probe template family executes on the actual ai-edge-litert interpreter."""
    fixture = fixture_for(op, tmp_path, meta=meta)
    result = CpuRunner().run(fixture, "cpu_xnnpack")
    assert result.status == "delegated", result.note
    assert result.max_abs_diff == 0.0


@requires_cpu
@pytest.mark.parametrize(
    ("op", "dtypes"),
    [
        ("CAST", ("float32",)),
        ("CAST", ("int64",)),
        ("GREATER", ("bool",)),
        ("SELECT_V2", ("float32",)),
    ],
)
def test_cpu_runner_measures_special_dtype_templates(
    op: str, dtypes: tuple[str, ...], tmp_path: Path
) -> None:
    """Signature dtypes name the OUTPUTS: casts, bool comparisons, and selects
    (bool condition input) all execute for real with the seeded feeds."""
    fixture = fixture_for(op, tmp_path, dtypes=dtypes, meta={"dynamic_shape": False, "rank": 2})
    result = CpuRunner().run(fixture, "cpu_xnnpack")
    assert result.status == "delegated", result.note
    assert result.max_abs_diff == 0.0


@requires_cpu
@pytest.mark.parametrize(
    ("op", "dtypes", "rank"),
    [
        # The graduated quantized-weight deferral executes for real: fp16
        # DEQUANTIZE and the int8-activation forms with canonical qparams.
        ("DEQUANTIZE", ("float32",), 2),
        ("CONV_2D", ("int8",), 4),
        ("DEPTHWISE_CONV_2D", ("int8",), 4),
        ("FULLY_CONNECTED", ("int8",), 2),
        ("AVERAGE_POOL_2D", ("int8",), 4),
        ("ADD", ("int8",), 4),
        ("PAD", ("int8",), 4),
        ("TRANSPOSE", ("int8",), 4),
        ("RESHAPE", ("int8",), 2),
    ],
)
def test_cpu_runner_measures_quantized_templates(
    op: str, dtypes: tuple[str, ...], rank: int, tmp_path: Path
) -> None:
    fixture = fixture_for(op, tmp_path, dtypes=dtypes,
                          meta={"dynamic_shape": False, "rank": rank})
    result = CpuRunner().run(fixture, "cpu_xnnpack")
    assert result.status == "delegated", result.note
    assert result.max_abs_diff == 0.0


@requires_cpu
def test_cpu_runner_reports_unrunnable_op_as_crash(tmp_path: Path) -> None:
    """No float kernel exists for SIN(int32): the honest cpu measurement is
    `crash`, never a guess."""
    fixture = fixture_for("SIN", tmp_path, dtypes=("int32",))
    result = CpuRunner().run(fixture, "cpu_xnnpack")
    assert result.status == "crash"
    assert result.max_abs_diff is None
    assert result.note


@pytest.mark.hardware
@pytest.mark.skipif(
    WebGpuMacRunner().availability() is not None,
    reason=f"webgpu_mac unavailable: {WebGpuMacRunner().availability()}",
)
def test_webgpu_mac_runner_measures_on_host_gpu(tmp_path: Path) -> None:
    runner = WebGpuMacRunner()
    fixture = fixture_for(
        "SOFTMAX", tmp_path, meta={"dynamic_shape": False, "rank": 2}, backend="webgpu_mac"
    )
    result = runner.run(fixture, "webgpu_mac")
    assert result.status in MATRIX_STATUSES
    if result.status in ("delegated", "incorrect"):
        assert result.max_abs_diff is not None
        # Evidence: the runtime log names the delegate that actually served.
        assert result.note and "delegate" in result.note


#: Both adb runners share one helper contract; every protocol test runs on both.
ADB_RUNNERS = [(AdbMlDriftRunner, "gpu_mldrift"), (NpuAdbRunner, "npu_qnn")]


@pytest.mark.parametrize(("runner_cls", "backend"), ADB_RUNNERS)
def test_adb_runner_unavailable_without_adb(
    monkeypatch: pytest.MonkeyPatch, runner_cls: type[AdbMlDriftRunner], backend: str
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert runner_cls().availability() == "adb not on PATH"


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["adb"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.mark.parametrize(("runner_cls", "backend"), ADB_RUNNERS)
def test_adb_runner_unavailable_without_device(
    monkeypatch: pytest.MonkeyPatch, runner_cls: type[AdbMlDriftRunner], backend: str
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/adb")
    runner = runner_cls()
    monkeypatch.setattr(
        runner, "_adb", lambda *args, timeout: _completed(stdout="List of devices attached\n\n")
    )
    assert runner.availability() == "no Android device connected via adb"


@pytest.mark.parametrize(("runner_cls", "backend"), ADB_RUNNERS)
def test_adb_runner_unavailable_without_helper(
    monkeypatch: pytest.MonkeyPatch, runner_cls: type[AdbMlDriftRunner], backend: str
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/adb")
    runner = runner_cls()

    def fake_adb(*args: str, timeout: int):
        if args[0] == "devices":
            return _completed(stdout="List of devices attached\nPIXEL8A\tdevice\n")
        return _completed(returncode=1)

    monkeypatch.setattr(runner, "_adb", fake_adb)
    reason = runner.availability()
    assert reason is not None and "on-device probe helper missing" in reason


@pytest.mark.parametrize(("runner_cls", "backend"), ADB_RUNNERS)
def test_adb_runner_parses_helper_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner_cls: type[AdbMlDriftRunner],
    backend: str,
) -> None:
    fixture = fixture_for("ABS", tmp_path, backend=backend)
    runner = runner_cls()

    def fake_adb(*args: str, timeout: int):
        if args[0] == "shell" and args[1] == runner.helper:
            return _completed(stdout='{"status": "incorrect", "max_abs_diff": 0.25}\n')
        return _completed()

    monkeypatch.setattr(runner, "_adb", fake_adb)
    result = runner.run(fixture, backend)
    assert result.status == "incorrect"
    assert result.max_abs_diff == 0.25


@pytest.mark.parametrize(("runner_cls", "backend"), ADB_RUNNERS)
def test_adb_runner_rejects_bad_helper_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner_cls: type[AdbMlDriftRunner],
    backend: str,
) -> None:
    fixture = fixture_for("ABS", tmp_path, backend=backend)
    runner = runner_cls()

    def fake_adb(*args: str, timeout: int):
        if args[0] == "shell" and args[1] == runner.helper:
            return _completed(stdout='{"status": "definitely-fine"}\n')
        return _completed()

    monkeypatch.setattr(runner, "_adb", fake_adb)
    with pytest.raises(RunnerError, match="helper output invalid"):
        runner.run(fixture, backend)

    monkeypatch.setattr(runner, "_adb", lambda *args, timeout: _completed(returncode=9))
    with pytest.raises(RunnerError, match="adb push failed"):
        runner.run(fixture, backend)


# --- webgpu_playwright (browser probes through the Phase 5 harness) ---------


def test_web_runner_registered_and_claims_web_backends() -> None:
    registry = runner_registry()
    runner = registry["webgpu_playwright"]
    assert runner.claims("wasm_xnnpack") and runner.claims("webgpu_mldrift")
    assert not runner.claims("webgpu_mac")
    # No other runner claims the web backends: selection is unambiguous.
    for name, other in registry.items():
        if name != "webgpu_playwright":
            assert not other.claims("wasm_xnnpack")
            assert not other.claims("webgpu_mldrift")


@requires_cpu
def test_web_runner_unavailable_without_harness(tmp_path: Path) -> None:
    reason = WebGpuPlaywrightRunner(sweep_dir=tmp_path / "nope").availability()
    assert reason is not None and "web/sweep" in reason


@requires_cpu
def test_page_inputs_refuse_unfeedable_dtype(tmp_path: Path) -> None:
    """LiteRT.js I/O is float32/int32; other dtypes are refused precisely,
    never coerced (coercion would break the byte-identical-inputs contract)."""
    from litert_compat.probe.runners import page_inputs, seeded_feeds

    fixture = fixture_for("SIN", tmp_path, dtypes=("float64",))
    with pytest.raises(RunnerError, match="float64"):
        page_inputs(seeded_feeds(fixture))


@requires_cpu
def test_interpret_probe_report_maps_every_outcome() -> None:
    import numpy as np

    from litert_compat.probe.runners import Tolerance, interpret_probe_report

    ref = [np.array([1.0, 2.0], dtype=np.float32)]
    tol = Tolerance()
    base = {"coreVersion": "2.5.3", "adapter": {"vendor": "apple", "architecture": "metal-3"}}

    ok = interpret_probe_report(
        {**base, "ok": True, "outputs": [[1.0, 2.0]], "fullyAccelerated": True},
        ref, "webgpu_mldrift", tol,
    )
    assert ok.status == "delegated" and ok.max_abs_diff == 0.0
    assert ok.note and "@litertjs/core 2.5.3" in ok.note and "metal-3" in ok.note

    off = interpret_probe_report(
        {**base, "ok": True, "outputs": [[1.5, 2.0]], "fullyAccelerated": True},
        ref, "webgpu_mldrift", tol,
    )
    assert off.status == "incorrect" and off.max_abs_diff == pytest.approx(0.5)

    partial = interpret_probe_report(
        {**base, "ok": True, "outputs": [[1.0, 2.0]], "fullyAccelerated": False},
        ref, "webgpu_mldrift", tol,
    )
    assert partial.status == "fallback" and "not fully accelerated" in (partial.note or "")

    refused = interpret_probe_report(
        {**base, "ok": False, "stage": "load", "error": "compile refused"},
        ref, "webgpu_mldrift", tol,
    )
    assert refused.status == "fallback" and "delegate refused" in (refused.note or "")

    wasm_load_fail = interpret_probe_report(
        {**base, "ok": False, "stage": "load", "error": "bad bytes"},
        ref, "wasm_xnnpack", tol,
    )
    assert wasm_load_fail.status == "crash"

    run_fail = interpret_probe_report(
        {**base, "ok": False, "stage": "run", "error": "boom"},
        ref, "webgpu_mldrift", tol,
    )
    assert run_fail.status == "crash"

    with pytest.raises(RunnerError, match="init"):
        interpret_probe_report(
            {**base, "ok": False, "stage": "init", "error": "no wasm"}, ref, "wasm_xnnpack", tol
        )


WEB_UNAVAILABLE = WebGpuPlaywrightRunner().availability()


@pytest.mark.hardware
@pytest.mark.skipif(
    WEB_UNAVAILABLE is not None, reason=f"webgpu_playwright unavailable: {WEB_UNAVAILABLE}"
)
def test_web_runner_measures_in_real_chromium(tmp_path: Path) -> None:
    """Real headless-Chromium probes on both web backends: the wasm CPU-class
    backend must execute the elementwise fixture; webgpu returns any honest
    matrix status with a real diff when numerics exist."""
    runner = WebGpuPlaywrightRunner()
    fixture = fixture_for(
        "ABS", tmp_path, meta={"dynamic_shape": False, "rank": 2}, backend="wasm_xnnpack"
    )
    wasm = runner.run(fixture, "wasm_xnnpack")
    assert wasm.status == "delegated", wasm.note
    assert wasm.max_abs_diff is not None and wasm.max_abs_diff <= 1e-5
    assert wasm.note and "@litertjs/core" in wasm.note

    webgpu = runner.run(fixture, "webgpu_mldrift")
    assert webgpu.status in MATRIX_STATUSES, webgpu.note
    if webgpu.status in ("delegated", "incorrect"):
        assert webgpu.max_abs_diff is not None
    assert webgpu.note and "@litertjs/core" in webgpu.note
