"""Probe suite: fixture generation, backend runners, release pipeline (Phase 8).

Probe fixtures are minimal single-op `.tflite` files built from complete lookup
signatures (linter `needs_probe` output, or `inferred` entries selected out of
a matrix snapshot). Runners execute them on real backends; `release-check`
orchestrates carry-forward → probe → re-measure → upgrade → diff → report.
Phase 8 adds no schema: probe output is matrix entries, reports reuse
`matrix diff` output.
"""

from litert_compat.probe.gen import ProbeFixture, build_probe_graph, generate_fixture
from litert_compat.probe.release import (
    ReleaseCheckError,
    check_snapshot,
    manifest_relint,
    release_check,
    render_report_markdown,
)
from litert_compat.probe.runners import (
    AdbMlDriftRunner,
    CpuRunner,
    Runner,
    RunnerError,
    RunnerResult,
    Tolerance,
    WebGpuMacRunner,
    runner_registry,
)
from litert_compat.probe.signatures import (
    ProbeSignature,
    UnprobeableSignatureError,
    make_signature,
    shape_meta_from_constraints,
    signature_from_entry,
    signatures_from_lint_report,
)

__all__ = [
    "AdbMlDriftRunner",
    "CpuRunner",
    "ProbeFixture",
    "ProbeSignature",
    "ReleaseCheckError",
    "Runner",
    "RunnerError",
    "RunnerResult",
    "Tolerance",
    "UnprobeableSignatureError",
    "WebGpuMacRunner",
    "build_probe_graph",
    "check_snapshot",
    "generate_fixture",
    "make_signature",
    "manifest_relint",
    "release_check",
    "render_report_markdown",
    "runner_registry",
    "shape_meta_from_constraints",
    "signature_from_entry",
    "signatures_from_lint_report",
]
