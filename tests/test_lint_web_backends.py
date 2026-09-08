"""Phases 6 + 13: the linter accepts the registered web AND NPU backend IDs
and returns honest `unknown`-dominated verdicts until op-level data exists
(the trap rule keeps those matrix snapshots empty — no runtime log or probe
result, no entry)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.lint.cli import app as lint_app

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
WEB_BACKENDS = ("wasm_xnnpack", "webgpu_mldrift", "webnn")
NPU_BACKENDS = ("npu_qnn", "npu_neuropilot")


def _json_report(output: str) -> dict:
    """Parse the leading JSON document from mixed CLI output.

    The runner mixes stderr into stdout, and the Phase 11 staleness warning
    fires whenever the example matrix's litert_version lags the autobumped
    releases registry — which is registry state, not what these tests assert.
    raw_decode reads the report and ignores the trailing warning line(s).
    """
    report, _ = json.JSONDecoder().raw_decode(output)
    return report


@pytest.mark.parametrize("backend", WEB_BACKENDS)
def test_lint_web_backend_runs_and_reports_unknown(backend: str) -> None:
    result = runner.invoke(
        lint_app,
        [
            str(EXAMPLES / "model_web_a_example.tflite"),
            "--matrix", str(EXAMPLES / f"matrix_{backend}_example.json"),
            "--backend", backend,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = _json_report(result.output)
    assert report["backend"] == backend
    assert report["litert_version"] == "2.5.3"
    assert report["findings"]  # every op surfaces, none silently passed
    assert all(f["status"] == "unknown" for f in report["findings"])
    assert report["summary"]["coverage_ops_pct"] == 0.0
    assert report["summary"]["claimed_ops"] == 0
    # honest unknowns become probe-ready signatures (anti-staleness §G.3):
    assert report["needs_probe"]
    assert all(sig["backend"] == backend for sig in report["needs_probe"])


@pytest.mark.parametrize("backend", WEB_BACKENDS)
def test_lint_web_backend_text_format_runs(backend: str) -> None:
    result = runner.invoke(
        lint_app,
        [
            str(EXAMPLES / "model_web_b_example.tflite"),
            "--matrix", str(EXAMPLES / f"matrix_{backend}_example.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "0.0%" in result.output


@pytest.mark.parametrize("backend", NPU_BACKENDS)
def test_lint_npu_backend_runs_and_reports_unknown(backend: str) -> None:
    """Phase 13 trap rule: NPU registration adds zero entries, so the linter
    reports honest `unknown` for every op and emits probe-ready signatures
    carrying the NPU backend id (the npu_adb runner's input shape)."""
    result = runner.invoke(
        lint_app,
        [
            str(EXAMPLES / "model_clean_example.tflite"),
            "--matrix", str(EXAMPLES / f"matrix_{backend}_example.json"),
            "--backend", backend,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = _json_report(result.output)
    assert report["backend"] == backend
    assert report["litert_version"] == "2.1.6"  # the version axis is native litert
    assert report["findings"]
    assert all(f["status"] == "unknown" for f in report["findings"])
    assert report["summary"]["coverage_ops_pct"] == 0.0
    assert report["needs_probe"]
    assert all(sig["backend"] == backend for sig in report["needs_probe"])


def test_lint_web_backend_mismatch_still_exits_2() -> None:
    """The one-snapshot-per-(backend x version) rule holds for web backends too."""
    result = runner.invoke(
        lint_app,
        [
            str(EXAMPLES / "model_web_a_example.tflite"),
            "--matrix", str(EXAMPLES / "matrix_wasm_xnnpack_example.json"),
            "--backend", "webgpu_mldrift",
        ],
    )
    assert result.exit_code == 2
