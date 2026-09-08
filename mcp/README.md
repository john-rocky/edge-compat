# edge-compat MCP server (Phase 14)

Status: **implemented** — the Phase 4 scaffold's tool surface, wired through
the official MCP Python SDK (`MCPServer`, the renamed FastMCP class) and
served over **stdio, local-first** (no hosting, no auth — spec Phase 14 Out).
Every tool is a thin pass-through into `litert_compat`; no logic lives here.

## Quickstart

```sh
# From the repo root (the `mcp` extra brings in the SDK):
uv run --extra mcp python mcp/server.py [--matrix-dir DIR] [--cards-dir DIR]
```

`data/matrix/` (the default `--matrix-dir`) is empty until the owner's first
real curation slice lands, so matrix-backed tools refuse honestly. To try the
tools against the example-provenance snapshot (example data stays out of
`data/matrix/` by design):

```sh
mkdir -p /tmp/litert-matrix && cp data/examples/matrix_example.json /tmp/litert-matrix/
uv run --extra mcp python mcp/server.py --matrix-dir /tmp/litert-matrix
```

### Client configs (one line each, run from the repo root)

- Claude Code: `claude mcp add edge-compat -- uv run --extra mcp python mcp/server.py`
- Gemini CLI: `gemini mcp add edge-compat uv run --extra mcp python mcp/server.py`

Both register a stdio server launched with the repo as working directory.
Add `--matrix-dir`/`--cards-dir` after `mcp/server.py` to serve other data.

## Tools

| Tool | What it does | Output governed by |
|---|---|---|
| `check_model(path, backend, litert_version?)` | Statically lint a `.tflite` model against the resolved matrix snapshot | `schemas/lint_report.schema.json` — identical to `edge-lint --json` |
| `check_op(op, dtype, backend, shape_meta?, litert_version?)` | Look up one op signature | verdict object `{status, matched_entry, reason, backend, litert_version}`; `matched_entry` is a `schemas/matrix.schema.json` entry, unchanged (`null` when `unknown`) |
| `suggest_rewrite(op, backend, condition?, litert_version?)` | Return matching rewrite hints | `{backend, litert_version, op, condition, hints}`; `hints` items are `schemas/matrix.schema.json` `rewrite_hints`, verbatim |
| `get_card(model_id)` | Return the model's card | `schemas/card.schema.json` (`cards/<model_id>/card.json`, unchanged) |
| `get_browser_status(model_id)` | Return the model's browser sweep statuses | `{model_id, browser}`; `browser` is the model's `cards/index.json` field, verbatim (per-backend statuses + `sweep_source`; `null` when never swept) |

Contract (spec Phases 4 + 14): tool outputs are the schema-validated JSON
objects produced by the earlier phases, **unchanged** — no reformatting, no
summarization (byte-equality is test-enforced). Matrix-backed responses embed
the resolved snapshot's `backend` + `litert_version` (Integration Rules §D),
so no output can present one backend's result as another's. `unknown` means
unmeasured — never a guess. Errors (unresolvable backend, unknown model id,
unreadable model) are structured MCP tool errors with precise reasons, not
prose results.

## Snapshot resolution (spec 14.1)

`litert_compat.matrix.resolve_snapshot` over `--matrix-dir`:

- exactly one snapshot for the backend → that one;
- multiple versions → the **highest numeric** version, unless the call pins
  `litert_version`;
- ambiguity (duplicate backend × version, or unpinned non-numeric versions)
  → a structured error, never a coin flip.

Snapshot identity comes from each file's own `backend`/`litert_version`
fields; file names are never trusted. Discovery is strict: every `*.json` in
the directory must be a snapshot, or the resolver refuses naming the file.

## Running / testing notes

This directory is named `mcp/`, the same as the SDK's package. Always run the
server by file path (as above). Never put the repo root on `sys.path` (no
`python -m mcp.server` from the root, no `__init__.py` here) — the local
directory would shadow the SDK package. Tests load `server.py` by file path
and drive the served process through a real `mcp.ClientSession` over stdio
(`tests/test_mcp_server.py`; they skip when the `mcp` extra is absent, and
fail loudly if the SDK import is ever shadowed).

Deferred: `get_npu_status(model_id)` (Phase 14 Ask-the-owner 2), hosting/
deployment/auth, telemetry (post-launch, opt-in per the proposal).
