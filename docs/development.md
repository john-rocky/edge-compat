# Repository layout and development

`DECISIONS` and `PROGRESS` citations on this page refer to the logs of the private lab repository.

## MCP server (Phase 14)

The dataset, served to AI agents: five tools over **stdio, local-first** —
`check_model` (full `edge-lint --json` report), `check_op` (matrix verdict,
entry verbatim), `suggest_rewrite` (curated `rewrite_hints`, verbatim),
`get_card` (`card.json` unchanged), `get_browser_status` (index browser
statuses + `sweep_source`). Outputs are the committed schema-validated objects,
never reformatted (byte-equality test-enforced); backend → snapshot resolution
picks the highest numeric `litert_version` unless pinned and refuses ambiguity.

```sh
uv run --extra mcp python mcp/server.py [--matrix-dir DIR] [--cards-dir DIR]
# Claude Code (from the repo root):
claude mcp add edge-compat -- uv run --extra mcp python mcp/server.py
```

Quickstart, tool table, and resolution policy: `mcp/README.md`.

## Repository layout

```
schemas/            # committed JSON Schemas — the public cross-phase contracts
data/matrix/        # real matrix snapshots, one per (backend x litert_version)
data/sweep/         # real browser sweep results; Phase 11 layout:
                    #   <litertjs_version>/<YYYY-MM-DD>/<model_id>.json, append-only
data/device_runs/   # real device-run snapshots; Phase 13 layout:
                    #   <runtime_version>/<YYYY-MM-DD>/<model_id>__<device>.json,
                    #   append-only, one runtime per snapshot directory
data/releases.json  # latest-known-release registry (staleness reference; autobump-maintained)
data/regression/    # catalog regression: bundles.json (own-bundle list) +
                    #   <runtime_version>/<YYYY-MM-DD>/<bundle_id>.json + SUMMARY.md (DECISIONS #205)
data/examples/      # example-provenance fixtures only; what tests run against
                    #   *.tflite fixtures are synthetic, built by
                    #   `python -m litert_compat.parser.fixtures` (byte-pinned in tests)
                    #   sweep/ holds example sweep results (schema-validated in tests)
                    #   device_runs/ holds example device-run snapshots
                    #   llm_samples/ holds REAL staged adapter-mapping samples
                    #     (read-only reference; never ingested — see its README)
data/probes/        # generated probe fixtures (probe gen default output; not committed)
data/transforms/    # owner-authored transform rules (examples live in
                    #   data/examples/transforms/ — agent-authored rules are
                    #   example-provenance only, spec §10.1)
src/litert_compat/  # library + CLIs
  matrix/           #   load / validate / query / import / diff / carry-forward
  parser/           #   .tflite flatbuffer reader + synthetic fixture builder
  lint/             #   edge-lint: classify, partition simulation, reports
  cards/            #   edge-card: build / build-all / index / enrich (+ adapters/)
  probe/            #   probe gen, backend runners incl. npu_adb, release-check (Phase 8/13)
  fix/              #   edge-fix: loader (mutate+reserialize), rule engine,
                    #   verify loop, rule authoring CLI (Phase 10)
  freshness/        #   Phase 11: releases registry, sweep snapshots, delta,
                    #   manual-verify register parsing
  device_runs/      #   Phase 13: device-run records, snapshot discovery,
                    #   gpu_audit / devicemark / compat_check adapters, CLI
tools/regression/   # catalog regression runner (list_bundles / run_regression / probe_one / summarize)
web/sweep/          # LiteRT.js browser sweep harness (TypeScript + Playwright)
site/               # demo-zoo static site generator (Phase 7; output in site/dist/, not committed)
cards/              # generated cards (committed example output; byte-pinned in tests)
mcp/                # MCP server (Phase 14; stdio, `mcp` extra — see mcp/README.md)
llms.txt            # generated discovery entry point (byte-pinned in tests)
DELTA_LOG.md        # generated per-release observed-changes log (Phase 11; never hand-edited)
MAINTENANCE.md      # manual-verify register (Phase 11; owner-maintained dates)
.github/workflows/  # deploy-zoo (7) + resweep / autobump / agentfix / reminders (11)
tests/              # incl. golden report files under tests/golden/
```

## Development

```sh
uv run pytest        # all tests run against example-provenance data
                     #   (real cpu-runner tests skip without `--extra runners`;
                     #    MCP client-session tests skip without `--extra mcp`;
                     #    GPU/device runner tests are hardware-marked and
                     #    auto-skip when the hardware is absent)
uv run ruff check .

cd web/sweep         # the browser-sweep lane (Node >= 20)
npm test             # build + unit tests
npm run typecheck && npm run lint

cd site              # the demo-zoo lane (Node >= 20)
npm test             # generator units + headless Playwright smoke
npm run typecheck && npm run lint
```
