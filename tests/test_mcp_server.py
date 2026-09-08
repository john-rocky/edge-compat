"""Phase 14 MCP server tests: direct tool calls + a real MCP client session.

Two layers, per the DoD:

- Unit: tools called directly on the file-path-loaded server module; outputs
  byte-equal the underlying schema-validated artifacts (committed card.json /
  index.json bytes, committed matrix entries) via the canonical serializer.
- Integration: the stdio server driven through a real `mcp.ClientSession`
  over a subprocess — list_tools, every tool, resolver pinning, and the
  structured-error paths.

Skips as a module when the `mcp` extra is not installed (the runners-extra
pattern). The module is loaded by file path — `mcp/` must never be importable
as a package or it would shadow the SDK (mcp/README.md); a guard below fails
loudly if that ever happens.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

try:
    import mcp as _mcp_sdk
except ImportError:  # pragma: no cover - environment without the extra
    pytest.skip("mcp SDK not installed (install the `mcp` extra)", allow_module_level=True)

if getattr(_mcp_sdk, "__file__", None) is None:  # pragma: no cover - defect guard
    raise RuntimeError(
        "`import mcp` resolved to a namespace package — the repo's mcp/ directory "
        "is shadowing the SDK; the repo root must never be on sys.path"
    )

from jsonschema import Draft202012Validator
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver.exceptions import ToolError

from litert_compat.lint.classify import classify_model
from litert_compat.lint.report import build_report, load_lint_report_schema
from litert_compat.matrix.canonical import canonical_dumps, load_json
from litert_compat.matrix.query import Matrix
from litert_compat.parser.reader import parse_tflite

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "mcp" / "server.py"
EXAMPLES = ROOT / "data" / "examples"
TOOL_NAMES = ["check_model", "check_op", "get_browser_status", "get_card", "suggest_rewrite"]


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("litert_compat_mcp_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def matrix_dir(tmp_path: Path) -> Path:
    """Snapshots dir: gpu_mldrift example + two wasm_xnnpack versions."""
    directory = tmp_path / "matrix"
    directory.mkdir()
    shutil.copy(EXAMPLES / "matrix_example.json", directory / "gpu_mldrift__example.json")
    shutil.copy(EXAMPLES / "matrix_wasm_xnnpack_example.json", directory / "wasm__2.5.3.json")
    newer = load_json(EXAMPLES / "matrix_wasm_xnnpack_example.json")
    newer["litert_version"] = "2.10.0"
    (directory / "wasm__2.10.0.json").write_text(canonical_dumps(newer), encoding="utf-8")
    return directory


@pytest.fixture()
def served(server, matrix_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "MATRIX_DIR", matrix_dir)
    return server


def gather_nd_entry() -> dict[str, Any]:
    doc = load_json(EXAMPLES / "matrix_example.json")
    (entry,) = [e for e in doc["entries"] if e["op"] == "GATHER_ND"]
    return entry


# --- Unit: byte-equality against the committed artifacts --------------------


def test_get_card_byte_equals_committed_card(server) -> None:
    committed = (ROOT / "cards" / "example-tiny-clean" / "card.json").read_text(encoding="utf-8")
    assert canonical_dumps(server.get_card("example-tiny-clean")) == committed


def test_get_card_unknown_id_is_tool_error(server) -> None:
    with pytest.raises(ToolError, match="no-such-model"):
        server.get_card("no-such-model")


def test_get_browser_status_byte_equals_committed_index_field(server) -> None:
    index_models = load_json(ROOT / "cards" / "index.json")["models"]
    (committed,) = [m["browser"] for m in index_models if m["id"] == "example-web-a"]
    result = server.get_browser_status("example-web-a")
    assert result["model_id"] == "example-web-a"
    assert canonical_dumps(result["browser"]) == canonical_dumps(committed)


def test_get_browser_status_never_swept_is_honest_null(server) -> None:
    assert server.get_browser_status("example-tiny-clean") == {
        "model_id": "example-tiny-clean",
        "browser": None,
    }


def test_get_browser_status_unknown_id_is_tool_error(server) -> None:
    with pytest.raises(ToolError, match="no-such-model"):
        server.get_browser_status("no-such-model")


def test_check_op_verdict_matches_library_and_committed_entry(served, matrix_dir: Path) -> None:
    result = served.check_op("GATHER_ND", "float32", "gpu_mldrift")
    expected = asdict(
        Matrix.load(matrix_dir / "gpu_mldrift__example.json").lookup("GATHER_ND", ["float32"])
    )
    assert result == expected
    assert canonical_dumps(result["matched_entry"]) == canonical_dumps(gather_nd_entry())
    assert (result["backend"], result["litert_version"]) == ("gpu_mldrift", "0.0.0-example")


def test_check_op_unknown_verdict_has_null_entry(served) -> None:
    result = served.check_op("ADD", "float32", "wasm_xnnpack", litert_version="2.5.3")
    assert result["status"] == "unknown"
    assert result["matched_entry"] is None
    assert (result["backend"], result["litert_version"]) == ("wasm_xnnpack", "2.5.3")


def test_check_op_unpinned_resolves_highest_numeric_version(served) -> None:
    assert served.check_op("ADD", "float32", "wasm_xnnpack")["litert_version"] == "2.10.0"


def test_check_op_no_snapshot_is_tool_error(served) -> None:
    with pytest.raises(ToolError, match="npu_qnn"):
        served.check_op("ADD", "float32", "npu_qnn")


def test_suggest_rewrite_hints_verbatim(served) -> None:
    result = served.suggest_rewrite("GATHER_ND", "gpu_mldrift", condition="step>1")
    assert (result["backend"], result["litert_version"]) == ("gpu_mldrift", "0.0.0-example")
    assert (result["op"], result["condition"]) == ("GATHER_ND", "step>1")
    committed = gather_nd_entry()["rewrite_hints"]
    assert canonical_dumps(result["hints"]) == canonical_dumps(committed)
    assert served.suggest_rewrite("GATHER_ND", "gpu_mldrift", condition="no such text")[
        "hints"
    ] == []


def test_check_model_report_equals_lint_json_and_validates(served, matrix_dir: Path) -> None:
    model_path = EXAMPLES / "model_mixed_example.tflite"
    result = served.check_model(str(model_path), "gpu_mldrift")

    snapshot = matrix_dir / "gpu_mldrift__example.json"
    matrix = Matrix.load(snapshot)
    model_bytes = model_path.read_bytes()
    parsed = parse_tflite(model_bytes)
    expected = build_report(
        classify_model(parsed, matrix),
        matrix,
        model_path=str(model_path),
        model_bytes=model_bytes,
        subgraph_count=len(parsed.subgraphs),
        matrix_path=str(snapshot),
    )
    assert canonical_dumps(result) == canonical_dumps(expected)
    errors = list(Draft202012Validator(load_lint_report_schema()).iter_errors(result))
    assert errors == []


def test_check_model_missing_file_is_tool_error(served) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        served.check_model(str(EXAMPLES / "nope.tflite"), "gpu_mldrift")


# --- Integration: a real MCP client session over stdio ----------------------


def session_calls(
    calls: list[tuple[str, dict[str, Any]]],
    matrix_dir: Path | None = None,
    list_tools: bool = False,
) -> tuple[list[str], list[Any]]:
    """Run `calls` in ONE stdio client session; -> (tool names, results)."""
    args = [str(SERVER_PATH)]
    if matrix_dir is not None:
        args += ["--matrix-dir", str(matrix_dir)]
    args += ["--cards-dir", str(ROOT / "cards")]
    params = StdioServerParameters(command=sys.executable, args=args, cwd=str(ROOT))

    async def run() -> tuple[list[str], list[Any]]:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            names: list[str] = []
            if list_tools:
                listed = await session.list_tools()
                names = sorted(tool.name for tool in listed.tools)
            results = [await session.call_tool(tool, arguments) for tool, arguments in calls]
            return names, results

    return asyncio.run(run())


def test_session_lists_exactly_the_five_tools(matrix_dir: Path) -> None:
    names, _ = session_calls([], matrix_dir=matrix_dir, list_tools=True)
    assert names == TOOL_NAMES


def test_session_matrix_tools_round_trip(served, matrix_dir: Path) -> None:
    model_path = EXAMPLES / "model_mixed_example.tflite"
    _, results = session_calls(
        [
            ("check_op", {"op": "GATHER_ND", "dtype": "float32", "backend": "gpu_mldrift"}),
            ("check_op", {"op": "ADD", "dtype": "float32", "backend": "wasm_xnnpack"}),
            (
                "check_op",
                {
                    "op": "ADD",
                    "dtype": "float32",
                    "backend": "wasm_xnnpack",
                    "litert_version": "2.5.3",
                },
            ),
            (
                "suggest_rewrite",
                {"op": "GATHER_ND", "backend": "gpu_mldrift", "condition": "step>1"},
            ),
            ("check_model", {"path": str(model_path), "backend": "gpu_mldrift"}),
        ],
        matrix_dir=matrix_dir,
    )
    assert all(not result.is_error for result in results)
    verdict, unpinned, pinned, rewrite, report = (result.structured_content for result in results)

    # Byte-for-byte the same objects the library serves (unit tests pin those
    # to the committed artifacts).
    assert canonical_dumps(verdict) == canonical_dumps(
        served.check_op("GATHER_ND", "float32", "gpu_mldrift")
    )
    assert unpinned["litert_version"] == "2.10.0"  # highest numeric wins
    assert pinned["litert_version"] == "2.5.3"  # unless pinned
    assert canonical_dumps(rewrite["hints"]) == canonical_dumps(
        gather_nd_entry()["rewrite_hints"]
    )
    assert canonical_dumps(report) == canonical_dumps(
        served.check_model(str(model_path), "gpu_mldrift")
    )


def test_session_card_tools_byte_equal_committed_artifacts(matrix_dir: Path) -> None:
    _, results = session_calls(
        [
            ("get_card", {"model_id": "example-tiny-clean"}),
            ("get_browser_status", {"model_id": "example-web-a"}),
        ],
        matrix_dir=matrix_dir,
    )
    card, browser = results
    assert not card.is_error and not browser.is_error
    committed_card = (ROOT / "cards" / "example-tiny-clean" / "card.json").read_text(
        encoding="utf-8"
    )
    assert canonical_dumps(card.structured_content) == committed_card
    index_models = load_json(ROOT / "cards" / "index.json")["models"]
    (committed_browser,) = [m["browser"] for m in index_models if m["id"] == "example-web-a"]
    assert canonical_dumps(browser.structured_content["browser"]) == canonical_dumps(
        committed_browser
    )


def test_session_errors_are_structured(matrix_dir: Path, tmp_path: Path) -> None:
    duplicated = tmp_path / "dup"
    duplicated.mkdir()
    shutil.copy(EXAMPLES / "matrix_example.json", duplicated / "a.json")
    shutil.copy(EXAMPLES / "matrix_example.json", duplicated / "b.json")

    _, results = session_calls(
        [
            ("get_card", {"model_id": "no-such-model"}),
            ("check_op", {"op": "ADD", "dtype": "float32", "backend": "gpu_mldrift"}),
        ],
        matrix_dir=duplicated,
    )
    unknown_card, duplicate_snapshot = results
    assert unknown_card.is_error
    assert "no card for model id 'no-such-model'" in unknown_card.content[0].text
    assert duplicate_snapshot.is_error
    assert "duplicate snapshots for backend 'gpu_mldrift'" in duplicate_snapshot.content[0].text


def test_session_empty_matrix_dir_refuses_honestly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _, results = session_calls(
        [("check_op", {"op": "ADD", "dtype": "float32", "backend": "gpu_mldrift"})],
        matrix_dir=empty,
    )
    (result,) = results
    assert result.is_error
    assert "no matrix snapshot for backend 'gpu_mldrift'" in result.content[0].text


def test_check_model_accepts_litertlm_bundle(served) -> None:
    """The MCP tool mirrors edge-lint's container handling: the embedded
    TFLite model is analyzed and the echoed path names the section."""
    bundle = EXAMPLES / "model_bundle_example.litertlm"
    result = served.check_model(str(bundle), "gpu_mldrift")
    assert "::TFLiteModel@section" in result["model"]["path"]
    errors = list(Draft202012Validator(load_lint_report_schema()).iter_errors(result))
    assert errors == []
