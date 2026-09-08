"""Pluggable backend runners: `run(fixture, backend) -> {status, max_abs_diff}`.

Five runners ship (spec §8.2 + Phase 13 + the graduated web runner): `cpu`
(ai-edge-litert interpreter — the numeric reference), `webgpu_mac`
(CompiledModel GPU on the host Mac, the `gpu_gate_mac.sh` pattern),
`gpu_mldrift_adb` (Android device over adb, gracefully "unavailable" without
a device), `npu_adb` (NPU accelerators over adb, same helper contract), and
`webgpu_playwright` (browser probes through the Phase 5 sweep harness on
headless Chromium — the designed path to lifting the Phase 6 trap rule;
gracefully "unavailable" without node/npm deps or the harness bundle).

Rules, in full force:
- Correctness is part of status — compiles-and-runs but exceeds the numeric
  tolerance is `incorrect`, not `delegated`.
- A runner never guesses: an unavailable backend produces no result and the
  affected entries stay `inferred` (no result, no entry). A runner that cannot
  produce a measurement raises RunnerError instead of returning one.

Inputs are seeded and deterministic (mulberry32, the same generator convention
as the Phase 5 sweep), drawn from [0.25, 1.25) so domain-restricted ops
(LOG, SQRT, RSQRT, DIV, POW) stay NaN-free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from litert_compat.parser.reader import TensorInfo, parse_tflite_file
from litert_compat.probe.gen import ProbeFixture

MATRIX_STATUSES = frozenset({"delegated", "partial", "fallback", "incorrect", "crash", "unknown"})

DEFAULT_ADB_HELPER = "/data/local/tmp/litert-probe-runner"
_ADB_HELPER_ENV = "LITERT_COMPAT_ADB_HELPER"
DEFAULT_NPU_ADB_HELPER = "/data/local/tmp/litert-npu-probe-runner"
_NPU_ADB_HELPER_ENV = "LITERT_COMPAT_NPU_ADB_HELPER"
_ADB_REMOTE_DIR = "/data/local/tmp/edge-compat-probes"


@dataclass(frozen=True)
class Tolerance:
    """Numeric pass bound vs the CPU reference: |out - ref| <= max(absolute,
    relative * |ref|), per element — the Phase 5 sweep's formula. Per-backend /
    per-dtype policy is an open Ask-the-owner item; these are the proposed
    defaults."""

    absolute: float = 1e-5
    relative: float = 1e-3


@dataclass(frozen=True)
class RunnerResult:
    status: str  # matrix status enum value
    max_abs_diff: float | None  # vs the CPU reference; None when no numerics exist
    note: str | None = None


class RunnerError(RuntimeError):
    """The runner could not produce a measurement. No result, no entry."""


class Runner(ABC):
    """One measurement backend. `availability()` returns None when usable,
    otherwise a human-readable reason (which the release report surfaces)."""

    name: str
    backends: frozenset[str]

    def claims(self, backend: str) -> bool:
        return backend in self.backends

    @abstractmethod
    def availability(self) -> str | None: ...

    @abstractmethod
    def run(self, fixture: ProbeFixture, backend: str) -> RunnerResult: ...


def _note(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".splitlines()[0]
    return text[:200]


def _mulberry32(seed: int) -> Callable[[], float]:
    state = seed & 0xFFFFFFFF

    def next_float() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return next_float


def _seed_for(fixture_id: str) -> int:
    return int.from_bytes(hashlib.sha256(fixture_id.encode("utf-8")).digest()[:4], "big")


def seeded_feeds_for(path: Path, seed_key: str) -> list[tuple[TensorInfo, Any]]:
    """Deterministic input arrays for a model's graph inputs, seeded from
    `seed_key`. Identical for every caller with the same key, so numeric
    comparisons are apples-to-apples (probe runners key on the fixture id;
    the edge-fix verify loop keys on the original model hash)."""
    import numpy as np

    parsed = parse_tflite_file(path)
    subgraph = parsed.subgraphs[0]
    rng = _mulberry32(_seed_for(seed_key))
    feeds: list[tuple[TensorInfo, Any]] = []
    for tensor_index in subgraph.inputs:
        tensor = subgraph.tensors[tensor_index]
        shape = tuple(max(dim, 1) for dim in (tensor.shape or ()))
        count = 1
        for dim in shape:
            count *= dim
        np_dtype = np.dtype(tensor.dtype)  # matrix dtype names are numpy names
        if np_dtype.kind == "f":
            values = [0.25 + rng() for _ in range(count)]
        elif np_dtype.kind == "b":
            values = [rng() >= 0.5 for _ in range(count)]
        else:
            values = [1 + int(rng() * 4) for _ in range(count)]
        feeds.append((tensor, np.array(values, dtype=np_dtype).reshape(shape)))
    return feeds


def seeded_feeds(fixture: ProbeFixture) -> list[tuple[TensorInfo, Any]]:
    return seeded_feeds_for(fixture.path, fixture.fixture_id)


def interpreter_outputs(path: Path, feeds: list[tuple[TensorInfo, Any]]) -> list[Any]:
    """Run a model on the ai-edge-litert interpreter with the given feeds.
    Raises on any interpreter failure — callers decide what a failure means."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    details = interpreter.get_input_details()
    if len(details) != len(feeds):
        raise RunnerError(
            f"{path}: interpreter reports {len(details)} inputs, "
            f"parser found {len(feeds)}"
        )
    for detail, (_tensor, array) in zip(details, feeds, strict=True):
        interpreter.resize_tensor_input(detail["index"], list(array.shape))
    interpreter.allocate_tensors()
    for detail, (_tensor, array) in zip(details, feeds, strict=True):
        interpreter.set_tensor(detail["index"], array)
    interpreter.invoke()
    return [interpreter.get_tensor(out["index"]) for out in interpreter.get_output_details()]


def cpu_reference_outputs(fixture: ProbeFixture) -> list[Any]:
    """Run the fixture on the ai-edge-litert interpreter with seeded inputs."""
    return interpreter_outputs(fixture.path, seeded_feeds(fixture))


def _max_abs_and_within(reference: list[Any], candidate: list[Any], tol: Tolerance) -> tuple[
    float, bool
]:
    import numpy as np

    max_abs = 0.0
    within = True
    for ref, out in zip(reference, candidate, strict=True):
        ref64 = np.asarray(ref, dtype=np.float64)
        out64 = np.asarray(out, dtype=np.float64).reshape(ref64.shape)
        if not np.all(np.isfinite(out64)):
            return float("inf"), False
        diff = np.abs(out64 - ref64)
        max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
        bound = np.maximum(tol.absolute, tol.relative * np.abs(ref64))
        within = within and bool(np.all(diff <= bound))
    return max_abs, within


class CpuRunner(Runner):
    """The numeric reference: ai-edge-litert interpreter (CPU/XNNPACK)."""

    name = "cpu"
    backends = frozenset({"cpu", "cpu_xnnpack"})

    def availability(self) -> str | None:
        try:
            import ai_edge_litert.interpreter  # noqa: F401
        except Exception as exc:
            return f"ai-edge-litert not importable ({_note(exc)}); install edge-compat[runners]"
        return None

    def run(self, fixture: ProbeFixture, backend: str) -> RunnerResult:
        try:
            cpu_reference_outputs(fixture)
        except Exception as exc:
            # The CPU backend has nothing to fall back to: failing to build or
            # execute the op IS the measured behavior.
            return RunnerResult(status="crash", max_abs_diff=None, note=_note(exc))
        return RunnerResult(status="delegated", max_abs_diff=0.0)


@contextlib.contextmanager
def _native_stderr_capture() -> Any:
    """Capture fd-2 output — the native runtime's logs — so the delegate that
    actually served can be recorded as evidence (the delegate behind
    `HardwareAccelerator.GPU` varies by wheel and platform)."""
    captured: dict[str, str] = {}
    try:
        saved = os.dup(2)
    except OSError:
        yield captured
        return
    try:
        with tempfile.TemporaryFile() as sink:
            os.dup2(sink.fileno(), 2)
            try:
                yield captured
            finally:
                os.dup2(saved, 2)
                sink.seek(0)
                captured["text"] = sink.read().decode("utf-8", "replace")
    finally:
        os.close(saved)


def _delegate_marker(log: str) -> str | None:
    """Deterministic one-liner naming the delegate the runtime log mentions."""
    lowered = log.lower()
    if "delegate_metal" in lowered:
        return "runtime log names the Metal delegate"
    if "webgpu" in lowered or "dawn" in lowered:
        return "runtime log names the WebGPU delegate"
    if "opencl" in lowered:
        return "runtime log names the OpenCL delegate"
    return None


class WebGpuMacRunner(Runner):
    """CompiledModel GPU on the host Mac (the `gpu_gate_mac.sh` /
    `pi5_gpu_probe.py` pattern). Mac host-GPU pass != Android ML Drift pass —
    this runner only ever claims `webgpu_mac`. Which delegate actually serves
    `HardwareAccelerator.GPU` varies by wheel (ai-edge-litert 2.1.6 on macOS
    uses the Metal delegate), so the runtime log is scanned and the serving
    delegate recorded in the result note as evidence."""

    name = "webgpu_mac"
    backends = frozenset({"webgpu_mac"})

    def __init__(self, tolerance: Tolerance | None = None) -> None:
        self.tolerance = tolerance or Tolerance()

    def availability(self) -> str | None:
        if sys.platform != "darwin":
            return "host is not macOS"
        try:
            import ai_edge_litert.compiled_model
            import ai_edge_litert.interpreter  # noqa: F401
        except Exception as exc:
            return f"ai-edge-litert not importable ({_note(exc)}); install edge-compat[runners]"
        return None

    def run(self, fixture: ProbeFixture, backend: str) -> RunnerResult:
        try:
            reference = cpu_reference_outputs(fixture)
        except Exception as exc:
            raise RunnerError(f"cpu reference failed: {_note(exc)}") from exc
        feeds = seeded_feeds(fixture)

        with _native_stderr_capture() as captured:
            error_status, fully, outputs, notes = self._compile_and_run(
                fixture, feeds, reference
            )
        marker = _delegate_marker(captured.get("text", ""))
        if marker:
            notes.append(marker)
        if error_status is not None:
            return RunnerResult(
                status=error_status, max_abs_diff=None, note="; ".join(notes) or None
            )

        max_abs, within = _max_abs_and_within(reference, outputs, self.tolerance)
        if fully is False:
            notes.insert(0, "graph not fully accelerated: the op executed on CPU fallback")
            return RunnerResult(status="fallback", max_abs_diff=max_abs, note="; ".join(notes))
        status = "delegated" if within else "incorrect"
        return RunnerResult(status=status, max_abs_diff=max_abs, note="; ".join(notes) or None)

    def _compile_and_run(
        self, fixture: ProbeFixture, feeds: list[tuple[TensorInfo, Any]], reference: list[Any]
    ) -> tuple[str | None, bool | None, list[Any] | None, list[str]]:
        """-> (error_status, fully_accelerated, outputs, notes)."""
        import numpy as np
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator

        notes: list[str] = []
        try:
            model = CompiledModel.from_file(str(fixture.path), HardwareAccelerator.GPU)
        except Exception as exc:
            # The delegate refused the graph: on-device this op falls back.
            notes.append(f"delegate refused the graph: {_note(exc)}")
            return "fallback", None, None, notes
        try:
            fully: bool | None
            try:
                fully = bool(model.is_fully_accelerated())
            except Exception as exc:
                fully = None
                notes.append(f"is_fully_accelerated query failed: {_note(exc)}")
            input_buffers = model.create_input_buffers(0)
            output_buffers = model.create_output_buffers(0)
            for buffer, (_tensor, array) in zip(input_buffers, feeds, strict=True):
                buffer.write(array.ravel())
            model.run_by_index(0, input_buffers, output_buffers)
            outputs = [
                np.array(buffer.read(int(ref.size), ref.dtype.type))
                for buffer, ref in zip(output_buffers, reference, strict=True)
            ]
        except Exception as exc:
            notes.append(_note(exc))
            return "crash", None, None, notes
        finally:
            model.close()
        return None, fully, outputs, notes


class AdbMlDriftRunner(Runner):
    """Android ML Drift over adb. Requires a connected device and an on-device
    probe helper (README: release pipeline) that prints one JSON line
    `{"status": ..., "max_abs_diff": ...}` — this runner never converts a
    missing helper or device into a guess."""

    name = "gpu_mldrift_adb"
    backends = frozenset({"gpu_mldrift"})

    def __init__(self, helper: str | None = None) -> None:
        self.helper = helper or os.environ.get(_ADB_HELPER_ENV, DEFAULT_ADB_HELPER)

    def _adb(self, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["adb", *args], capture_output=True, text=True, timeout=timeout, check=False
        )

    def availability(self) -> str | None:
        if shutil.which("adb") is None:
            return "adb not on PATH"
        try:
            devices = self._adb("devices", timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"adb devices failed: {_note(exc)}"
        attached = [
            line for line in devices.stdout.splitlines()[1:] if line.strip().endswith("\tdevice")
        ]
        if not attached:
            return "no Android device connected via adb"
        try:
            helper = self._adb("shell", "test", "-x", self.helper, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"adb shell failed: {_note(exc)}"
        if helper.returncode != 0:
            return (
                f"on-device probe helper missing at {self.helper} "
                f"(set ${_ADB_HELPER_ENV}; see README: release pipeline)"
            )
        return None

    def run(self, fixture: ProbeFixture, backend: str) -> RunnerResult:
        remote = f"{_ADB_REMOTE_DIR}/{fixture.fixture_id}.tflite"
        try:
            self._adb("shell", "mkdir", "-p", _ADB_REMOTE_DIR, timeout=15)
            push = self._adb("push", str(fixture.path), remote, timeout=60)
            if push.returncode != 0:
                raise RunnerError(f"adb push failed: {push.stderr.strip()[:200]}")
            result = self._adb(
                "shell", self.helper, "--model", remote, "--backend", backend, timeout=300
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RunnerError(f"adb invocation failed: {_note(exc)}") from exc
        if result.returncode != 0:
            raise RunnerError(
                f"on-device helper exited {result.returncode}: {result.stderr.strip()[:200]}"
            )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        try:
            payload = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError as exc:
            raise RunnerError(f"on-device helper output is not JSON: {_note(exc)}") from exc
        if not isinstance(payload, dict) or payload.get("status") not in MATRIX_STATUSES:
            tail = lines[-1][:200] if lines else ""
            raise RunnerError(f"on-device helper output invalid: {tail}")
        diff = payload.get("max_abs_diff")
        return RunnerResult(
            status=payload["status"],
            max_abs_diff=float(diff) if diff is not None else None,
            note=payload.get("note"),
        )


class NpuAdbRunner(AdbMlDriftRunner):
    """NPU accelerators over adb (Phase 13) — the same one-JSON-line helper
    contract as `gpu_mldrift_adb`, pointed at an NPU-capable on-device helper
    (vendor SDK linkage is the owner's build; this runner never guesses).
    Claims the NPU backend IDs registered in matrix.schema.json 1.3. NPU
    probes may require per-SoC AOT compilation: signatures the helper cannot
    compile come back as a RunnerError (surfaced as unprobeable-with-reason
    by the release pipeline), never as a fabricated status."""

    name = "npu_adb"
    backends = frozenset({"npu_qnn", "npu_neuropilot"})

    def __init__(self, helper: str | None = None) -> None:
        super().__init__(
            helper or os.environ.get(_NPU_ADB_HELPER_ENV, DEFAULT_NPU_ADB_HELPER)
        )


_WEB_SWEEP_DIR_ENV = "LITERT_COMPAT_WEB_SWEEP_DIR"
_PAGE_DTYPES = frozenset({"float32", "int32"})


def _default_web_sweep_dir() -> Path:
    """The repo's web/sweep checkout (absent in installed wheels — the
    availability check reports that honestly instead of guessing)."""
    override = os.environ.get(_WEB_SWEEP_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "web" / "sweep"


def page_inputs(feeds: list[tuple[TensorInfo, Any]]) -> list[dict[str, Any]]:
    """Serialize seeded feeds for the in-page harness (`FixtureInput` shape).

    LiteRT.js tensor I/O is float32/int32 only; any other dtype is a precise
    refusal (RunnerError), never a coerced tensor — coercion would compare the
    browser against inputs the CPU reference never saw.
    """
    inputs: list[dict[str, Any]] = []
    for tensor, array in feeds:
        if tensor.dtype not in _PAGE_DTYPES:
            raise RunnerError(
                f"LiteRT.js tensor I/O is float32/int32 only; cannot feed "
                f"dtype {tensor.dtype}"
            )
        inputs.append(
            {
                "shape": [int(dim) for dim in array.shape],
                "dtype": tensor.dtype,
                "data": [float(v) if tensor.dtype == "float32" else int(v) for v in array.ravel()],
            }
        )
    return inputs


def interpret_probe_report(
    payload: dict[str, Any],
    reference: list[Any],
    backend: str,
    tolerance: Tolerance,
) -> RunnerResult:
    """Map one driver report line onto the runner contract.

    webgpu: a load refusal is `fallback` (the delegate rejected the graph, as
    on-device the op would run on CPU), a run failure is `crash`, partial
    acceleration is `fallback`. wasm_xnnpack is itself the CPU-class backend —
    it has nothing to fall back to, so load/run failures are `crash` (the
    CpuRunner semantics). Numerics decide delegated vs incorrect either way.
    """
    import numpy as np

    notes: list[str] = [f"@litertjs/core {payload.get('coreVersion', '?')}"]
    adapter = payload.get("adapter")
    if isinstance(adapter, dict) and backend != "wasm_xnnpack":
        vendor = adapter.get("vendor") or "?"
        arch = adapter.get("architecture") or "?"
        notes.append(f"webgpu adapter {vendor}/{arch}")
    notes.extend(str(line) for line in (payload.get("evidence") or [])[:2])

    if not payload.get("ok"):
        stage = payload.get("stage")
        error = str(payload.get("error") or "unknown error")[:200]
        if stage in ("init", "probe"):
            raise RunnerError(f"browser environment failed at {stage}: {error}")
        if stage == "load":
            if backend == "wasm_xnnpack":
                notes.append(f"load failed: {error}")
                return RunnerResult(status="crash", max_abs_diff=None, note="; ".join(notes))
            notes.insert(1, f"delegate refused the graph: {error}")
            return RunnerResult(status="fallback", max_abs_diff=None, note="; ".join(notes))
        notes.append(f"run failed: {error}")
        return RunnerResult(status="crash", max_abs_diff=None, note="; ".join(notes))

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(reference):
        raise RunnerError(
            f"driver returned {len(outputs) if isinstance(outputs, list) else 'no'} "
            f"outputs, reference has {len(reference)}"
        )
    candidate = [
        np.asarray(out, dtype=np.asarray(ref).dtype).reshape(np.asarray(ref).shape)
        for out, ref in zip(outputs, reference, strict=True)
    ]
    max_abs, within = _max_abs_and_within(reference, candidate, tolerance)
    if backend != "wasm_xnnpack" and payload.get("fullyAccelerated") is False:
        notes.insert(1, "graph not fully accelerated: the op executed on WASM fallback")
        return RunnerResult(status="fallback", max_abs_diff=max_abs, note="; ".join(notes))
    status = "delegated" if within else "incorrect"
    return RunnerResult(status=status, max_abs_diff=max_abs, note="; ".join(notes))


class WebGpuPlaywrightRunner(Runner):
    """Browser probes through the Phase 5 sweep harness (headless Chromium) —
    the designed path to lifting the Phase 6 trap rule: op-level web matrix
    entries measured for real, never derived from model-level sweeps.

    The version axis for the snapshots this runner populates is the
    `@litertjs/core` version, NOT an ai-edge-litert version: pass it as
    release-check's `--to-litert-version`, and cross-check it against the
    measured core version this runner records in every result note.

    Numeric honesty: the exact seeded feeds used for the CPU reference are
    materialized into the job file and fed to the page verbatim (the harness
    page never generates probe inputs); outputs come back verbatim and the
    comparison runs here against the same reference.
    """

    name = "webgpu_playwright"
    backends = frozenset({"wasm_xnnpack", "webgpu_mldrift"})

    _ACCELERATORS: ClassVar[dict[str, str]] = {
        "wasm_xnnpack": "wasm",
        "webgpu_mldrift": "webgpu",
    }

    def __init__(
        self,
        tolerance: Tolerance | None = None,
        sweep_dir: Path | None = None,
        timeout_s: int = 300,
    ) -> None:
        self.tolerance = tolerance or Tolerance()
        self.sweep_dir = sweep_dir or _default_web_sweep_dir()
        self.timeout_s = timeout_s

    def availability(self) -> str | None:
        try:
            import ai_edge_litert.interpreter  # noqa: F401
        except Exception as exc:
            return (
                f"ai-edge-litert not importable ({_note(exc)}); the CPU reference "
                "needs edge-compat[runners]"
            )
        if not (self.sweep_dir / "src" / "node" / "probe.ts").is_file():
            return f"web/sweep harness not found at {self.sweep_dir} (set ${_WEB_SWEEP_DIR_ENV})"
        if shutil.which("node") is None:
            return "node not on PATH (web/sweep needs Node >= 20)"
        if not (self.sweep_dir / "node_modules" / "playwright").is_dir():
            return f"web/sweep dependencies not installed (run `npm ci` in {self.sweep_dir})"
        if not (self.sweep_dir / "dist" / "page" / "harness.js").is_file():
            return f"harness bundle missing (run `npm run build` in {self.sweep_dir})"
        return None

    def run(self, fixture: ProbeFixture, backend: str) -> RunnerResult:
        try:
            reference = cpu_reference_outputs(fixture)
        except Exception as exc:
            raise RunnerError(f"cpu reference failed: {_note(exc)}") from exc
        inputs = page_inputs(seeded_feeds(fixture))
        job = {
            "modelPath": str(fixture.path.resolve()),
            "accelerator": self._ACCELERATORS[backend],
            "inputs": inputs,
            "timeoutMs": self.timeout_s * 1000,
        }
        payload = self._invoke_driver(job)
        return interpret_probe_report(payload, reference, backend, self.tolerance)

    def _invoke_driver(self, job: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="litert-web-probe-") as tmp:
            job_path = Path(tmp) / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        "node",
                        "--experimental-strip-types",
                        str(Path("src") / "node" / "probe.ts"),
                        "--job",
                        str(job_path),
                    ],
                    cwd=self.sweep_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s + 60,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RunnerError(f"probe driver invocation failed: {_note(exc)}") from exc
        if proc.returncode != 0:
            raise RunnerError(
                f"probe driver exited {proc.returncode}: {proc.stderr.strip()[:200]}"
            )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        try:
            payload = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError as exc:
            raise RunnerError(f"probe driver output is not JSON: {_note(exc)}") from exc
        if not isinstance(payload, dict):
            raise RunnerError("probe driver printed no report line")
        return payload


def runner_registry(tolerance: Tolerance | None = None) -> dict[str, Runner]:
    """The shipped runners, keyed by CLI name."""
    return {
        "cpu": CpuRunner(),
        "webgpu_mac": WebGpuMacRunner(tolerance),
        "gpu_mldrift_adb": AdbMlDriftRunner(),
        "npu_adb": NpuAdbRunner(),
        "webgpu_playwright": WebGpuPlaywrightRunner(tolerance),
    }
