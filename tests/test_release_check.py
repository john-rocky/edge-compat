"""release-check: the carry-forward → probe → re-measure → upgrade → diff →
report cycle, driven by a deterministic fake runner (spec §8.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import entry, make_doc
from litert_compat.cli import app
from litert_compat.matrix.canonical import load_json, write_canonical
from litert_compat.probe.gen import ProbeFixture
from litert_compat.probe.release import (
    ReleaseCheckError,
    release_check,
    render_report_markdown,
)
from litert_compat.probe.runners import Runner, RunnerError, RunnerResult, Tolerance

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"

cli = CliRunner()


class FakeRunner(Runner):
    """Deterministic canned results, keyed by op. `unknown` and RunnerError
    are expressible; everything else defaults to delegated/0.0."""

    name = "fake"

    def __init__(
        self,
        backend: str = "gpu_mldrift",
        results: dict[str, RunnerResult] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        self.backends = frozenset({backend})
        self.results = results or {}
        self.errors = errors or {}

    def availability(self) -> str | None:
        return None

    def run(self, fixture: ProbeFixture, backend: str) -> RunnerResult:
        op = fixture.signature.op
        if op in self.errors:
            raise RunnerError(self.errors[op])
        return self.results.get(op, RunnerResult(status="delegated", max_abs_diff=0.0))


class UnavailableRunner(FakeRunner):
    name = "fake_unavailable"

    def availability(self) -> str | None:
        return "device is on fire"


def probe_evidence(fixture_id: str) -> dict[str, str]:
    return {
        "source_model": f"probe:{fixture_id}",
        "litert_version": "1.2.0",
        "date": "2026-08-01",
    }


def run_check(tmp_path: Path, doc: dict, runners: list[Runner], **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshot = tmp_path / f"{doc['backend']}__{doc['litert_version']}.json"
    write_canonical(doc, snapshot)
    out_dir = kwargs.pop("out_dir", tmp_path / "out")
    return release_check(
        [snapshot],
        to_litert_version=kwargs.pop("to_litert_version", "1.3.0"),
        generated_at=kwargs.pop("generated_at", "2026-08-10"),
        lint_report_docs=kwargs.pop("lint_report_docs", []),
        runners=runners,
        out_dir=out_dir,
        tolerance=Tolerance(),
        **kwargs,
    ), out_dir


def base_doc(entries: list[dict]) -> dict:
    return make_doc(entries, litert_version="1.2.0", generated_at="2026-08-01")


def test_cycle_probe_confirms_and_upgrades(tmp_path: Path) -> None:
    """measured → (carry-forward) inferred → (probe) measured again. When the
    new runtime behaves identically, release-check honestly reports NOTHING
    changed — the §G loop closing cleanly."""
    doc = base_doc(
        [
            entry(
                op="NEG",
                provenance="measured",
                dtypes=["float32"],
                evidence=probe_evidence("NEG__float32__c383a423"),
            )
        ]
    )
    report, out_dir = run_check(tmp_path, doc, [FakeRunner()])
    backend = report["backends"][0]
    assert backend["counts"] == {"upgraded": 1}
    assert backend["remaining_inferred"] == 0
    assert not report["has_changes"]
    new_doc = load_json(out_dir / "gpu_mldrift__1.3.0.json")
    (new_entry,) = new_doc["entries"]
    assert new_entry["provenance"] == "measured"
    assert new_entry["evidence"]["litert_version"] == "1.3.0"
    assert new_entry["evidence"]["source_model"].startswith("probe:NEG__float32__")
    assert "derived_from" not in new_entry["evidence"]


def test_incorrect_numerics_become_a_status_transition(tmp_path: Path) -> None:
    doc = base_doc(
        [
            entry(
                op="SOFTMAX",
                provenance="measured",
                dtypes=["float32"],
                evidence=probe_evidence("SOFTMAX__float32__c383a423"),
            )
        ]
    )
    fake = FakeRunner(
        results={"SOFTMAX": RunnerResult(status="incorrect", max_abs_diff=0.34)}
    )
    report, out_dir = run_check(tmp_path, doc, [fake])
    assert report["has_changes"]
    diff = report["backends"][0]["diff"]
    assert diff["status_transitions"] == [
        {
            "op": "SOFTMAX",
            "dtypes": ["float32"],
            "constraints": {},
            "from": "delegated",
            "to": "incorrect",
        }
    ]
    (new_entry,) = load_json(out_dir / "gpu_mldrift__1.3.0.json")["entries"]
    assert new_entry["status"] == "incorrect"
    assert new_entry["provenance"] == "measured"


def test_conflict_with_full_model_measurement_is_never_overwritten(tmp_path: Path) -> None:
    doc = base_doc(
        [
            entry(
                op="SQRT",
                status="fallback",
                provenance="measured",
                dtypes=["float32"],
                evidence={
                    "source_model": "example-model-x",
                    "litert_version": "1.2.0",
                    "date": "2026-08-01",
                },
            )
        ]
    )
    report, out_dir = run_check(tmp_path, doc, [FakeRunner()])  # probe says delegated
    backend = report["backends"][0]
    assert backend["counts"] == {"conflict": 1}
    (record,) = backend["probes"]
    assert record["disposition"] == "conflict"
    assert "full-model measurement" in record["note"]
    (new_entry,) = load_json(out_dir / "gpu_mldrift__1.3.0.json")["entries"]
    assert new_entry["status"] == "fallback"  # original verdict kept
    assert new_entry["provenance"] == "inferred"  # staleness stays visible
    assert new_entry["evidence"]["source_model"] == "example-model-x"
    assert new_entry["evidence"]["derived_from"]["litert_version"] == "1.2.0"


def test_unknown_result_is_not_a_measurement(tmp_path: Path) -> None:
    doc = base_doc(
        [
            entry(
                op="TANH",
                provenance="measured",
                dtypes=["float32"],
                evidence=probe_evidence("TANH__float32__c383a423"),
            )
        ]
    )
    fake = FakeRunner(results={"TANH": RunnerResult(status="unknown", max_abs_diff=None)})
    report, out_dir = run_check(tmp_path, doc, [fake])
    assert report["backends"][0]["counts"] == {"unknown_result": 1}
    (new_entry,) = load_json(out_dir / "gpu_mldrift__1.3.0.json")["entries"]
    assert new_entry["provenance"] == "inferred"


def test_unavailable_backend_stays_inferred_and_is_remaining_work(tmp_path: Path) -> None:
    doc = base_doc(
        [
            entry(
                op="NEG",
                provenance="measured",
                dtypes=["float32"],
                evidence=probe_evidence("NEG__float32__c383a423"),
            )
        ]
    )
    report, _out_dir = run_check(tmp_path, doc, [UnavailableRunner()])
    backend = report["backends"][0]
    assert backend["runner"] is None
    assert backend["runner_reason"] == "fake_unavailable: device is on fire"
    assert backend["counts"] == {"no_runner": 1}
    assert backend["remaining_inferred"] == 1
    # The un-re-measured release IS a semantic change: staleness became visible.
    assert report["has_changes"]
    assert backend["diff"]["provenance_transitions"] == [
        {
            "op": "NEG",
            "dtypes": ["float32"],
            "constraints": {},
            "from": "measured",
            "to": "inferred",
        }
    ]


def test_no_runner_claims_backend(tmp_path: Path) -> None:
    doc = base_doc([entry(op="NEG", provenance="inferred", dtypes=["float32"])])
    report, _ = run_check(tmp_path, doc, [FakeRunner(backend="webgpu_mac")])
    backend = report["backends"][0]
    assert backend["runner"] is None
    assert backend["runner_reason"] == "no runner claims backend 'gpu_mldrift'"


def test_needs_probe_creates_entries_and_flags_duplicates(tmp_path: Path) -> None:
    doc = base_doc(
        [
            entry(
                op="RESHAPE",
                provenance="measured",
                dtypes=["float32"],
                constraints={"dynamic_shape": False, "rank": 2},
                evidence=probe_evidence("RESHAPE__float32__eb9e96e4"),
            )
        ]
    )
    lint_doc = json.loads((GOLDEN_DIR / "split.json").read_text(encoding="utf-8"))
    report, out_dir = run_check(tmp_path, doc, [FakeRunner()], lint_report_docs=[lint_doc])
    backend = report["backends"][0]
    # RESHAPE signature == the existing entry's identity -> duplicate;
    # SOFTMAX float32 rank-2 is new -> created.
    assert backend["counts"] == {"created": 1, "duplicate": 1, "upgraded": 1}
    new_doc = load_json(out_dir / "gpu_mldrift__1.3.0.json")
    created = [e for e in new_doc["entries"] if e["op"] == "SOFTMAX"]
    assert created == [
        {
            "op": "SOFTMAX",
            "dtypes": ["float32"],
            "constraints": {"dynamic_shape": False, "rank": 2},
            "status": "delegated",
            "provenance": "measured",
            "evidence": {
                "source_model": "probe:SOFTMAX__float32__eb9e96e4",
                "litert_version": "1.3.0",
                "date": "2026-08-10",
            },
        }
    ]


def test_needs_probe_backend_without_snapshot_is_reported(tmp_path: Path) -> None:
    doc = base_doc([])
    lint_doc = {
        "needs_probe": [
            {
                "backend": "webgpu_mldrift",
                "op": "RESHAPE",
                "dtypes": ["float32"],
                "shape_meta": {"dynamic_shape": False, "rank": 2},
            }
        ]
    }
    report, _ = run_check(tmp_path, doc, [FakeRunner()], lint_report_docs=[lint_doc])
    assert report["needs_probe_unmatched_backends"] == ["webgpu_mldrift"]


def test_runner_error_and_unprobeable_are_remaining_work(tmp_path: Path) -> None:
    doc = base_doc(
        [
            entry(op="ABS", provenance="inferred", dtypes=["float32"]),
            entry(
                op="MEAN",
                provenance="inferred",
                dtypes=["float32"],
                constraints={"keep_dims": True},
            ),
            entry(op="LSTM", provenance="inferred", dtypes=["float32"]),
        ]
    )
    fake = FakeRunner(errors={"ABS": "device fell off the desk"})
    report, out_dir = run_check(tmp_path, doc, [fake])
    backend = report["backends"][0]
    assert backend["counts"] == {"runner_error": 1, "unprobeable": 2}
    assert backend["remaining_inferred"] == 3
    new_doc = load_json(out_dir / "gpu_mldrift__1.3.0.json")
    assert all(e["provenance"] == "inferred" for e in new_doc["entries"])


def test_same_version_and_duplicate_backend_are_usage_errors(tmp_path: Path) -> None:
    doc = base_doc([])
    snapshot = tmp_path / "snap.json"
    write_canonical(doc, snapshot)
    with pytest.raises(ReleaseCheckError, match="already at litert_version"):
        release_check(
            [snapshot],
            to_litert_version="1.2.0",
            generated_at="2026-08-10",
            lint_report_docs=[],
            runners=[],
            out_dir=tmp_path / "out",
            tolerance=Tolerance(),
        )
    other = tmp_path / "snap2.json"
    write_canonical(doc, other)
    with pytest.raises(ReleaseCheckError, match="share backend"):
        release_check(
            [snapshot, other],
            to_litert_version="1.3.0",
            generated_at="2026-08-10",
            lint_report_docs=[],
            runners=[],
            out_dir=tmp_path / "out",
            tolerance=Tolerance(),
        )


def golden_inputs() -> tuple[dict, dict]:
    """The golden scenario: every disposition appears once, on example-shaped
    data, with the committed golden lint report supplying needs_probe."""
    doc = base_doc(
        [
            entry(
                op="NEG",
                provenance="measured",
                dtypes=["float32"],
                evidence=probe_evidence("NEG__float32__c383a423"),
            ),
            entry(
                op="SOFTMAX",
                provenance="measured",
                dtypes=["float16"],
                evidence=probe_evidence("SOFTMAX__float16__c383a423"),
            ),
            entry(
                op="SQRT",
                status="fallback",
                provenance="measured",
                dtypes=["float32"],
                evidence={
                    "source_model": "example-model-x",
                    "litert_version": "1.2.0",
                    "date": "2026-08-01",
                },
            ),
            entry(
                op="MEAN",
                provenance="inferred",
                dtypes=["float32"],
                constraints={"keep_dims": True},
                evidence={"derived_from": {"litert_version": "1.1.0", "date": "2026-07-01"}},
            ),
            entry(op="LSTM", provenance="inferred", dtypes=["float32"]),
            entry(op="TANH", provenance="inferred", dtypes=["float32"]),
            entry(op="ABS", provenance="inferred", dtypes=["float32"]),
            entry(op="CONV_2D", provenance="vendor_doc", dtypes=["float32"]),
        ]
    )
    lint_doc = json.loads((GOLDEN_DIR / "split.json").read_text(encoding="utf-8"))
    return doc, lint_doc


def golden_runner() -> FakeRunner:
    return FakeRunner(
        results={
            "SOFTMAX": RunnerResult(status="incorrect", max_abs_diff=0.34),
            "TANH": RunnerResult(status="unknown", max_abs_diff=None),
        },
        errors={"ABS": "device fell off the desk"},
    )


def run_golden(tmp_path: Path) -> tuple[dict, Path]:
    doc, lint_doc = golden_inputs()
    return run_check(
        tmp_path,
        doc,
        [golden_runner()],
        lint_report_docs=[lint_doc],
        manifest=EXAMPLES / "cards_manifest_example.csv",
    )


def test_golden_release_report(tmp_path: Path) -> None:
    report, out_dir = run_golden(tmp_path)
    from litert_compat.matrix.canonical import canonical_dumps

    assert canonical_dumps(report) == (GOLDEN_DIR / "release_report.json").read_text(
        encoding="utf-8"
    )
    assert render_report_markdown(report) == (GOLDEN_DIR / "release_report.md").read_text(
        encoding="utf-8"
    )
    assert (out_dir / "gpu_mldrift__1.3.0.json").exists()


def test_release_check_is_byte_deterministic(tmp_path: Path) -> None:
    from litert_compat.matrix.canonical import canonical_dumps

    report_a, out_a = run_golden(tmp_path / "run1")
    report_b, out_b = run_golden(tmp_path / "run2")
    assert canonical_dumps(report_a) == canonical_dumps(report_b)
    assert render_report_markdown(report_a) == render_report_markdown(report_b)
    assert (out_a / "gpu_mldrift__1.3.0.json").read_bytes() == (
        out_b / "gpu_mldrift__1.3.0.json"
    ).read_bytes()


def test_cli_release_check_nothing_changed_exit_0(
    tmp_path: Path, example_matrix_path: Path
) -> None:
    """Example matrix: all `example` provenance, nothing to probe — the new
    snapshot is semantically identical and the exit code says so."""
    out_dir = tmp_path / "out"
    args = [
        "release-check",
        str(example_matrix_path),
        "--to-litert-version", "0.0.1-example",
        "--generated-at", "2026-08-10",
        "-o", str(out_dir),
        "--runner", "cpu",
    ]
    result = cli.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "no semantic differences" in result.output
    first = (out_dir / "release_report.json").read_bytes()
    assert (out_dir / "release_report.md").exists()
    assert (out_dir / "gpu_mldrift__0.0.1-example.json").exists()
    result = cli.invoke(app, args)
    assert result.exit_code == 0
    assert (out_dir / "release_report.json").read_bytes() == first


def test_cli_release_check_unknown_runner_exit_2(
    tmp_path: Path, example_matrix_path: Path
) -> None:
    result = cli.invoke(
        app,
        [
            "release-check", str(example_matrix_path),
            "--to-litert-version", "9.9.9",
            "-o", str(tmp_path),
            "--runner", "quantum_annealer",
        ],
    )
    assert result.exit_code == 2
    assert "unknown runner" in result.output
