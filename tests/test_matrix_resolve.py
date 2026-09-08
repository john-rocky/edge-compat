"""Resolver policy (spec 14.1) and the rewrite-hints query (Phase 14).

Policy under test: exactly one snapshot per backend -> that one; multiple
versions -> highest numeric version unless pinned; ambiguity -> error, never
a coin flip. Hints are verbatim pass-throughs of committed matrix data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_compat.matrix.canonical import load_json
from litert_compat.matrix.query import Matrix
from litert_compat.matrix.resolve import (
    SnapshotResolutionError,
    numeric_version,
    resolve_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_snapshot(directory: Path, name: str, backend: str, litert_version: str) -> Path:
    path = directory / name
    path.write_text(
        json.dumps({"backend": backend, "litert_version": litert_version, "entries": []}),
        encoding="utf-8",
    )
    return path


def test_numeric_version() -> None:
    assert numeric_version("2.1.6") == (2, 1, 6)
    assert numeric_version("2.10.0") == (2, 10, 0)
    assert numeric_version("0.0.0-example") is None
    assert numeric_version("") is None


def test_single_snapshot_resolves(tmp_path: Path) -> None:
    path = write_snapshot(tmp_path, "a.json", "gpu_mldrift", "0.0.0-example")
    assert resolve_snapshot(tmp_path, "gpu_mldrift") == path


def test_highest_numeric_version_wins_numerically(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "old.json", "wasm_xnnpack", "2.5.3")
    newest = write_snapshot(tmp_path, "new.json", "wasm_xnnpack", "2.10.0")
    # 2.10.0 > 2.5.3 numerically although "2.10.0" < "2.5.3" lexicographically.
    assert resolve_snapshot(tmp_path, "wasm_xnnpack") == newest


def test_pin_selects_exact_version(tmp_path: Path) -> None:
    pinned = write_snapshot(tmp_path, "old.json", "wasm_xnnpack", "2.5.3")
    write_snapshot(tmp_path, "new.json", "wasm_xnnpack", "2.10.0")
    assert resolve_snapshot(tmp_path, "wasm_xnnpack", "2.5.3") == pinned


def test_pin_missing_version_is_error_listing_available(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "a.json", "wasm_xnnpack", "2.5.3")
    with pytest.raises(SnapshotResolutionError, match=r"2\.5\.3"):
        resolve_snapshot(tmp_path, "wasm_xnnpack", "9.9.9")


def test_pin_resolves_non_numeric_among_multiple(tmp_path: Path) -> None:
    example = write_snapshot(tmp_path, "a.json", "gpu_mldrift", "0.0.0-example")
    write_snapshot(tmp_path, "b.json", "gpu_mldrift", "2.1.6")
    assert resolve_snapshot(tmp_path, "gpu_mldrift", "0.0.0-example") == example


def test_unpinned_non_numeric_among_multiple_is_ambiguity_error(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "a.json", "gpu_mldrift", "0.0.0-example")
    write_snapshot(tmp_path, "b.json", "gpu_mldrift", "2.1.6")
    with pytest.raises(SnapshotResolutionError, match="pin litert_version"):
        resolve_snapshot(tmp_path, "gpu_mldrift")


def test_duplicate_backend_version_is_error(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "a.json", "gpu_mldrift", "2.1.6")
    write_snapshot(tmp_path, "b.json", "gpu_mldrift", "2.1.6")
    with pytest.raises(SnapshotResolutionError, match="duplicate"):
        resolve_snapshot(tmp_path, "gpu_mldrift")
    # The duplicate is a data defect even when the pin names it.
    with pytest.raises(SnapshotResolutionError, match="duplicate"):
        resolve_snapshot(tmp_path, "gpu_mldrift", "2.1.6")


def test_unknown_backend_lists_available(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "a.json", "gpu_mldrift", "2.1.6")
    write_snapshot(tmp_path, "b.json", "wasm_xnnpack", "2.5.3")
    with pytest.raises(SnapshotResolutionError, match="gpu_mldrift, wasm_xnnpack"):
        resolve_snapshot(tmp_path, "npu_qnn")


def test_missing_directory_is_error(tmp_path: Path) -> None:
    with pytest.raises(SnapshotResolutionError, match="does not exist"):
        resolve_snapshot(tmp_path / "nope", "gpu_mldrift")


def test_invalid_json_file_is_error_naming_file(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SnapshotResolutionError, match=r"broken\.json"):
        resolve_snapshot(tmp_path, "gpu_mldrift")


def test_non_snapshot_json_is_error_naming_file(tmp_path: Path) -> None:
    (tmp_path / "stray.json").write_text('{"models": []}', encoding="utf-8")
    with pytest.raises(SnapshotResolutionError, match=r"stray\.json"):
        resolve_snapshot(tmp_path, "gpu_mldrift")


# --- Matrix.rewrite_hints ---------------------------------------------------


@pytest.fixture(scope="module")
def example_matrix() -> Matrix:
    return Matrix.load(REPO_ROOT / "data" / "examples" / "matrix_example.json")


def test_rewrite_hints_verbatim_identity(example_matrix: Matrix) -> None:
    hints = example_matrix.rewrite_hints("GATHER_ND")
    committed = [
        hint
        for entry in load_json(REPO_ROOT / "data" / "examples" / "matrix_example.json")["entries"]
        if entry["op"] == "GATHER_ND"
        for hint in entry.get("rewrite_hints", [])
    ]
    assert hints == committed
    # Pass-through by identity: the returned objects ARE the loaded entries'
    # hint dicts, not copies — nothing is reformatted.
    entry_hints = [
        hint
        for entry in example_matrix.doc["entries"]
        if entry["op"] == "GATHER_ND"
        for hint in entry.get("rewrite_hints", [])
    ]
    assert all(
        hint is committed_hint for hint, committed_hint in zip(hints, entry_hints, strict=True)
    )


def test_rewrite_hints_condition_narrows_case_insensitively(example_matrix: Matrix) -> None:
    assert example_matrix.rewrite_hints("GATHER_ND", "STEP>1") == example_matrix.rewrite_hints(
        "GATHER_ND"
    )
    assert example_matrix.rewrite_hints("GATHER_ND", "no such condition text") == []


def test_rewrite_hints_unknown_or_hintless_op_is_empty(example_matrix: Matrix) -> None:
    assert example_matrix.rewrite_hints("NO_SUCH_OP") == []
    # CUMSUM has an entry in the example matrix but no rewrite_hints.
    assert example_matrix.rewrite_hints("CUMSUM") == []
