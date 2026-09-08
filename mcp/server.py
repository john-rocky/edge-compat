"""edge-compat MCP server — Phase 14 implementation.

Serves the toolkit's schema-validated JSON to MCP agent clients (Claude Code,
Gemini CLI first) over stdio. Five tools: `check_model`, `check_op`,
`suggest_rewrite`, `get_card`, `get_browser_status`. Every tool is a thin
pass-through into `litert_compat` — no logic is reimplemented here (Phase 4
rule) — and outputs are the schema-validated JSON objects from earlier phases,
unchanged: no reformatting, no summarization. Matrix-backed responses embed the
resolved snapshot's `backend` + `litert_version` (Integration Rules §D), so no
output can present one backend's result as another's.

Backend -> snapshot resolution follows `litert_compat.matrix.resolve_snapshot`
(spec 14.1): one snapshot per backend -> that one; multiple versions -> the
highest numeric version unless the call pins `litert_version`; ambiguity is a
structured error, never a coin flip. `data/matrix/` is empty until the owner's
first real curation slice lands, so a default-directory server refuses matrix
lookups honestly; point `--matrix-dir` at example snapshots to try the tools.

Run (requires the `mcp` extra): `uv run --extra mcp python mcp/server.py
[--matrix-dir DIR] [--cards-dir DIR]`. Quickstart and client configs:
mcp/README.md. Caution: this directory is named `mcp/` — always run the server
by file path and never put the repo root on `sys.path`, or this directory
shadows the SDK package.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from litert_compat.cards.index import CardIndexError, collect_cards
from litert_compat.lint.classify import classify_model
from litert_compat.lint.report import build_report
from litert_compat.matrix.canonical import load_json
from litert_compat.matrix.query import Matrix, MatrixLookupTieError
from litert_compat.matrix.resolve import SnapshotResolutionError, resolve_snapshot
from litert_compat.matrix.validate import MatrixValidationError
from litert_compat.parser.litertlm import (
    LitertlmParseError,
    extract_tflite_model,
    is_litertlm,
)
from litert_compat.parser.reader import TfliteParseError, parse_tflite

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = REPO_ROOT / "data" / "matrix"
CARDS_DIR = REPO_ROOT / "cards"

server = MCPServer(
    "edge-compat",
    instructions=(
        "Delegate compatibility data for LiteRT, from measured conversions. "
        "Verdicts are matrix lookups — `unknown` means unmeasured, never a "
        "guess — and every matrix-backed response names the snapshot's "
        "backend + litert_version it came from."
    ),
)


def _load_matrix(backend: str, litert_version: str | None) -> Matrix:
    """Resolve and load the snapshot serving `backend`; ToolError on refusal."""
    try:
        snapshot = resolve_snapshot(MATRIX_DIR, backend, litert_version)
    except SnapshotResolutionError as exc:
        raise ToolError(str(exc)) from exc
    try:
        return Matrix.load(snapshot)
    except (MatrixValidationError, ValueError) as exc:
        raise ToolError(f"{snapshot}: {exc}") from exc


@server.tool()
def check_model(
    path: str,
    backend: str,
    litert_version: str | None = None,
) -> dict[str, Any]:
    """Statically lint the model at `path` for `backend` delegation.

    Accepts a `.tflite` model or a LiteRT-LM `.litertlm` bundle (detected by
    magic; the embedded TFLite model is extracted and linted — delegate-level
    analysis only, and the echoed model path names the analyzed section).
    Output: the full lint report object, governed by
    `schemas/lint_report.schema.json` — identical to `edge-lint --json`
    against the resolved snapshot (coverage, per-op findings with matched-entry
    provenance, partitions, blocking-op ranking, probe-ready unknown-op
    signatures). `litert_version` pins a snapshot version when several serve
    the backend.
    """
    matrix = _load_matrix(backend, litert_version)
    model_file = Path(path)
    if not model_file.is_file():
        raise ToolError(f"model file {path!r} does not exist")
    model_bytes = model_file.read_bytes()
    model_path_echo = path
    if is_litertlm(model_bytes):
        try:
            model_bytes, section = extract_tflite_model(model_bytes)
        except LitertlmParseError as exc:
            raise ToolError(f"{path}: {exc}") from exc
        suffix = f"::TFLiteModel@section{section.index}"
        if section.model_type:
            suffix += f"({section.model_type})"
        model_path_echo += suffix
    try:
        parsed = parse_tflite(model_bytes)
    except TfliteParseError as exc:
        raise ToolError(f"{path}: {exc}") from exc
    try:
        classified = classify_model(parsed, matrix)
    except MatrixLookupTieError as exc:
        raise ToolError(f"matrix data defect: {exc}") from exc
    return build_report(
        classified,
        matrix,
        model_path=model_path_echo,
        model_bytes=model_bytes,
        subgraph_count=len(parsed.subgraphs),
        matrix_path=str(matrix.source),
    )


@server.tool()
def check_op(
    op: str,
    dtype: str,
    backend: str,
    shape_meta: dict[str, Any] | None = None,
    litert_version: str | None = None,
) -> dict[str, Any]:
    """Look up one op signature in `backend`'s matrix snapshot.

    `op` is an exact TFLite builtin operator name (e.g. `FULLY_CONNECTED`).
    Output: the `Matrix.lookup` verdict, serialized field-for-field —
    `{status, matched_entry, reason, backend, litert_version}` where
    `matched_entry` is a `schemas/matrix.schema.json` entry object, unchanged
    (null when the verdict is `unknown`). `unknown` means unmeasured — never
    a guess.
    """
    matrix = _load_matrix(backend, litert_version)
    try:
        return asdict(matrix.lookup(op, [dtype], shape_meta))
    except MatrixLookupTieError as exc:
        raise ToolError(f"matrix data defect: {exc}") from exc


@server.tool()
def suggest_rewrite(
    op: str,
    backend: str,
    condition: str | None = None,
    litert_version: str | None = None,
) -> dict[str, Any]:
    """Rewrite hints for `op` from `backend`'s matrix snapshot.

    Output: `{backend, litert_version, op, condition, hints}` where `hints`
    are `rewrite_hints` items exactly as committed in the matrix
    (`schemas/matrix.schema.json`), verbatim — hints are curated data, never
    generated. `condition` narrows by case-insensitive substring against the
    owning entry's `conditions` text and the hint text. An empty list means
    no curated hint exists.
    """
    matrix = _load_matrix(backend, litert_version)
    return {
        "backend": matrix.backend,
        "litert_version": matrix.litert_version,
        "op": op,
        "condition": condition,
        "hints": matrix.rewrite_hints(op, condition),
    }


@server.tool()
def get_card(model_id: str) -> dict[str, Any]:
    """The model's card, i.e. `cards/<model_id>/card.json`.

    Output: the card object, schema-validated against
    `schemas/card.schema.json` and returned unchanged. `model_id` is matched
    against the validated card catalog (never used to build a filesystem
    path).
    """
    try:
        cards = collect_cards(CARDS_DIR)
    except CardIndexError as exc:
        raise ToolError(str(exc)) from exc
    for card_id, card in cards:
        if card_id == model_id:
            return card
    raise ToolError(f"no card for model id {model_id!r} under {CARDS_DIR}")


@server.tool()
def get_browser_status(model_id: str) -> dict[str, Any]:
    """The model's browser (LiteRT.js) sweep statuses from the cards index.

    Output: `{model_id, browser}` where `browser` is the model's entry from
    the committed `cards/index.json`, verbatim: per-backend statuses
    (`load_failed | run_failed | output_mismatch | fallback | pass`) plus
    `sweep_source`, the repo-relative path of the raw sweep record. `browser`
    is null when the model has never been swept — reported honestly, never
    inferred.
    """
    index_path = CARDS_DIR / "index.json"
    if not index_path.is_file():
        raise ToolError(f"no cards index at {index_path}; run `edge-card index`")
    try:
        index_doc = load_json(index_path)
    except ValueError as exc:
        raise ToolError(f"{index_path}: not valid JSON: {exc}") from exc
    for model in index_doc.get("models", []):
        if model.get("id") == model_id:
            return {"model_id": model_id, "browser": model.get("browser")}
    raise ToolError(f"no model id {model_id!r} in {index_path}")


def main() -> None:
    """Serve the five tools to MCP clients over stdio."""
    global MATRIX_DIR, CARDS_DIR
    parser = argparse.ArgumentParser(description="edge-compat MCP server (stdio).")
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=MATRIX_DIR,
        help="Matrix snapshots directory (default: data/matrix/).",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=CARDS_DIR,
        help="Cards directory holding <id>/card.json + index.json (default: cards/).",
    )
    args = parser.parse_args()
    MATRIX_DIR = args.matrix_dir
    CARDS_DIR = args.cards_dir
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
