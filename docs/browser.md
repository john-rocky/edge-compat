# Browser

`DECISIONS` and `PROGRESS` citations on this page refer to the logs of the private lab repository.

## `web/sweep` — LiteRT.js browser sweep harness

The Phase 5 instrument (TypeScript + Playwright, the repo's one non-Python
lane): batch-measures real `.tflite` models on LiteRT.js in headless Chromium
— loads / runs / full-delegation / output-match-vs-CPU / latency p50, per
model × backend (`wasm_xnnpack`, `webgpu_mldrift`, `webnn` behind
`--experimental`) — writing one
[`schemas/sweep_result.schema.json`](../schemas/sweep_result.schema.json)-valid
JSON per model. Every record carries a mandatory `env` block (browser,
headless flag, WebGPU adapter, `@litertjs/core` version, machine label):
headless desktop Chromium is not a user's phone, and results are only ever
presented with their environment. Sweep results are model-level facts only —
op-level matrix entries are never derived from them (Phase 6 trap rule).

```sh
cd web/sweep && npm install && npx playwright install chromium
npm run sweep -- --catalog ../../data/examples/web_catalog_example.csv --out out/
```

Flags, the catalog manifest contract (`data/web_catalog.csv`), and the
determinism guarantee are documented in [web/sweep/README.md](../web/sweep/README.md).
The committed example results under `data/examples/sweep/` were produced by
this harness from the example catalog (which includes a deliberately broken
entry — a model that fails is a result, not an error).

## `site/` — demo zoo (Phase 7)

The public human-facing layer, generated from the machine-facing data: a
sortable model × backend × status × latency table (env-labeled; passing AND
failing models — a model that does not run in the browser is information),
plus a live demo page per passing model that loads the `.tflite` from its
official source URL at runtime and runs it in the visitor's own browser via
`@litertjs/core`, showing the backend actually in use and live latency next
to the env-labeled sweep numbers. Vanilla TypeScript, no framework, no
weights ever hosted (test- and CI-guarded).

```sh
cd site && npm install
npm run build     # cards/index.json + card.json (+ sweep records) -> site/dist/
npm test          # unit tests + headless Playwright smoke on a demo page
```

Demo pages exist only for models whose sweep results pass (`runs` on ≥ 1
backend — the honesty gate). Deployment is `deploy-zoo.yml` (manual
`workflow_dispatch`; deploys to GitHub Pages only on green build + smoke +
weights guard). Details: [site/README.md](../site/README.md). The site is deployed
from the public mirror; the lab repo only builds it as a CI check.

### "Check your model" playground (Phase 9)

`site/dist/check/` — a page where a visitor drops their own `.tflite` and
gets sweep-equivalent results (loads / runs / full delegation / output match
vs the wasm reference / latency p50, with failure classes explained in plain
language) measured live on both `wasm_xnnpack` and `webgpu_mldrift`, labeled
with *their* environment. **The model never leaves the browser** — read, parsed,
and run locally; the smoke test asserts no post-drop network request could
carry model-derived bytes. The page also parses the model's operator inventory
locally (operator names only — no graph analysis in TS) and cross-references
it against the web-backend matrix snapshots exported by the build
(`--matrix-dir`, default `data/matrix/`): verdicts come only from the exported
matrix JSON, so with the browser columns still honest-empty every op reads
`unknown` — by design, per the trap rule.

## Browser backends (Phase 6)

Matrix schema `1.1` registers the browser backend IDs — `wasm_xnnpack`,
`webgpu_mldrift`, `webnn` — with the **`@litertjs/core` version as the version
axis** (same one-snapshot-per-(backend × version) rule, web edition).

**The trap rule (data integrity, spelled out):** op-level matrix entries are
**never derived from model-level sweep results**. "Model X fully delegated"
does not license `delegated` entries for every op in X (op behavior varies
with dtype/shape context), and "model X fell back" attributes nothing to any
specific op. Op-level web entries may be created only when a runtime log names
a specific op, with the log captured as `evidence`. Until then the browser
matrix snapshots stay empty (honest empty examples under
`data/examples/matrix_{wasm_xnnpack,webgpu_mldrift,webnn}_example.json`), and
`edge-lint --backend webgpu_mldrift` (and peers) runs fine while honestly
reporting `unknown`-dominated verdicts — every unknown becomes a probe-ready
`needs_probe` signature for the future web runner (see "Deferred" in
PROGRESS.md).
