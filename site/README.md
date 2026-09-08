# `site/` — demo zoo static site generator (Phases 7 + 9)

Generates the public human-facing layer from the machine-facing data: a
sortable compatibility-and-latency table, one live demo page per model that
actually runs in the browser, and the "Check your model" playground
(`check/`) where visitors run their own `.tflite` locally. Vanilla
TypeScript + `@litertjs/core`; no framework, no server side.

```sh
cd site && npm install
npm run build          # -> site/dist/ (not committed)
npm test               # unit tests + headless Playwright smoke (needs `npx playwright install chromium`)
```

## What gets generated

```
dist/
  index.html           # model × backend × status × latency (env-labeled), sortable;
                       #   passing AND failing models — a model that does not run
                       #   in the browser is information, not something to hide
  models/<id>/         # demo page per PASSING model (honesty gate below)
  check/               # "Check your model" playground (Phase 9)
  matrix/<backend>.json# web-backend matrix snapshots, byte-verbatim (playground input;
                       #   absent while data/matrix/ has no web snapshots — the default)
  cards/<id>/          # card.json + CARD.md, byte-verbatim copies
  sweeps/<id>.json     # raw sweep records behind the cards, byte-verbatim
  assets/              # site.css, bundled demo.js / sort.js / playground.js,
                       #   @litertjs/core WASM
```

Inputs: `cards/index.json` + per-model `card.json` (+ each card's linked
`sweep_source` file for sample-input specs and sweep config). Statuses are
recomputed with the DECISIONS #62 vocabulary and cross-checked against
`cards/index.json` — the site can never disagree with the committed catalog.

## Honesty rules

- **Demo pages only for models that pass**: at least one backend with
  `runs: true` in the sweep results. Models with fixture-file input specs are
  also skipped (webcam/vision demo pages are deferred). Every skip and its
  reason is printed by the build and shown in the index.
- **No weights, ever**: demo pages fetch the `.tflite` from its official
  source URL at runtime (`source_url` as-is when it ends in `.tflite`,
  otherwise Hugging-Face-style `<source_url>/resolve/main/<artifact>`).
  A test and a CI guard assert no model binary lands in the output.
- **Env labels everywhere**: sweep numbers are always shown with the machine
  label, browser, and `@litertjs/core` version that produced them; the live
  run is labeled as the visitor's own hardware.
- **Example-provenance data is bannered** on every page that shows it.

## The playground (`check/`, Phase 9)

- **Hard privacy rule (spec 9.1): the visitor's model never leaves the
  browser.** The dropped `.tflite` is read, parsed, and run locally; no
  upload, no beacon containing model bytes or hashes of them. The smoke test
  asserts that after a model is dropped, every network request is a bodyless
  same-origin GET for a static asset — a shape that cannot carry
  model-derived bytes. Stated on the page; it is the feature.
- **Live results** per backend (`wasm_xnnpack` reference first, then
  `webgpu_mldrift`): the sweep's fields — loads (failure class with a
  plain-language explanation, incl. the WASM memory ceiling), runs (random
  seeded inputs built from the model's own input signature; v1 has no
  user-supplied tensors), full delegation with console warnings captured
  verbatim as evidence, output match vs the wasm reference with the sweep's
  exact comparator and tolerances, latency p50 after warmup — all labeled
  with the visitor's environment (their user agent, their GPU adapter).
- **Op cross-reference**: the operator inventory is parsed in-page
  (operator names only; the hand-written reader mirrors the Python parser's
  vocabulary, enforced by a parity test) and cross-referenced against the
  web-backend matrix snapshots exported at build time. Verdicts come ONLY
  from that exported matrix JSON — matching entries verbatim (status,
  dtypes, conditions, rewrite hints), `unknown` otherwise. No linter logic
  is reimplemented in TS (parity note, spec 9.2).
- **Matrix export**: `--matrix-dir` (default `data/matrix/`) is scanned for
  snapshots whose `backend` is a web ID (`wasm_xnnpack`, `webgpu_mldrift`,
  `webnn`); each is copied byte-verbatim to `dist/matrix/<backend>.json`.
  Two snapshots for one backend fail the build (version-selection policy is
  an open owner decision). Zero snapshots — today's reality — is the honest
  default: the page says so and every op reads `unknown`.

## Flags

```
npm run build -- [--out <dir>] [--repo-url <url>] [--cookbook-url <url>]
                 [--matrix-dir <dir>] [--releases <path>] [--lineup <file>]
```

CTA links default to omitted: the cookbook has no public remote yet. The "Build this
yourself" prompt text works without them.

`--lineup <file>` (Phase 12.2) builds the curated launch site: a text file of
model ids, one per line (`#` comments allowed) — demo pages are generated
ONLY for the listed ids, while the index table still shows every model,
passing and failing (the honesty rule is the launch's selling point). Unknown
ids and ids that fail the demo honesty gate are build errors, never silent
skips. The default build is unchanged.

## Branding and reproduce prompts (Phase 12)

Every page footer renders the disclosure line and the affiliation
notice from `src/shared/branding.ts`, which a test holds in exact
parity with `src/litert_compat/branding.py` — the post-approval rename and
the final disclosure wording are one edit per lane. Each page also carries a
"Reproduce this with an agent" copy-button (index → sweep the example
catalog; demo page → one-row catalog for that model; playground →
`edge-lint` locally); prompts name real commands only, and
`edge-compat release-gate --site-dist site/dist` verifies the disclosure
line on every generated page.

## Determinism

Same inputs + same dependency versions ⇒ byte-identical `dist/`, proven by a
double-build test. No timestamps, no randomness; the only dates on any page
come from the data itself.

## GitHub Pages notes

- Served without COOP/COEP (Pages cannot set headers): `crossOriginIsolated`
  is false, so LiteRT.js picks a non-threaded WASM variant. The smoke test
  serves the same way on purpose. JSPI is attempted first, with automatic
  fallback.
- The demo auto-runs on WebGPU when the visitor's browser supports it and
  falls back to WASM on failure; buttons re-run on either backend.
- `.github/workflows/deploy-zoo.yml` (manual `workflow_dispatch` only)
  builds, tests, guards, and deploys `site/dist` to Pages **only on green**.

## Measurement

Repo-stats only for now: no client-side counter, no cookies, no PII. The
`measure()` no-op in `src/page/demo.ts` marks the three event points (page
load, backend used, CTA copy) for a future owner-chosen counter.
