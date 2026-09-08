# edge-compat browser sweep harness

Batch measurement of real `.tflite` models on LiteRT.js in headless Chromium
(Playwright). For each catalog entry × backend it records: whether the model
**loads**, whether one inference **runs**, whether the graph was **fully
delegated** (WebGPU/WebNN), the **output match** against the `wasm_xnnpack`
CPU reference, and **latency p50** — one
[`schemas/sweep_result.schema.json`](../../schemas/sweep_result.schema.json)-valid
JSON file per model.

**Honesty rule:** headless desktop Chromium is not a user's phone. Every
record carries a mandatory `env` block (browser + version, headless flag, OS,
WebGPU adapter, `@litertjs/core` version, machine label); results are only
ever presented with it.

**Model-level results only:** op-level matrix entries are never derived from
sweep records (Phase 6 trap rule). "Model X fully delegated" licenses no
per-op verdict.

## Prerequisites

- Node ≥ 20 (`npm install` in this directory)
- Playwright Chromium: `npx playwright install chromium`

## Commands

```bash
npm run sweep -- --catalog ../../data/examples/web_catalog_example.csv --out out/   # run a sweep
npm test          # build + unit tests (node:test)
npm run typecheck # tsc --noEmit (strict)
npm run lint      # eslint (type-checked rules)
```

### Sweep flags

| Flag | Default | Meaning |
|---|---|---|
| `--catalog <path>` | required | catalog manifest CSV (format below) |
| `--out <dir>` | required | output dir; writes one `<model_id>.json` per model |
| `--experimental` | off | include the `webnn` backend |
| `--headed` | off | run a visible browser (recorded in `env.headless`) |
| `--machine-label <s>` | `unlabeled` | `env.machine_label` on every record |
| `--date <YYYY-MM-DD>` | today | record date (pass explicitly for reproducible output) |
| `--provenance <p>` | `measured` | `measured` \| `example` — fixture sweeps must use `example` |
| `--warmup <n>` | 3 | warmup runs per backend |
| `--runs <n>` | 10 | timed runs per backend (p50 over these) |
| `--tolerance-abs <x>` | 1e-5 | output-match absolute tolerance |
| `--tolerance-rel <x>` | 1e-3 | output-match relative tolerance |
| `--seed <n>` | 42 | fixture-input PRNG seed |
| `--timeout-ms <n>` | 60000 | per-step timeout per model × backend |
| `--cache-dir <dir>` | off | cache URL-sourced model downloads (content-addressed by source URL), staged one model at a time right before its sweep; a re-run of an unchanged catalog downloads nothing. A failed download falls back to the direct URL so the honest `fetch_failed` result is still recorded |
| `--cache-max-bytes <n>` | unbounded | with `--cache-dir`: around each model, evict least-recently-used cached models until the directory holds at most `n` bytes. `0` keeps nothing between models — the CI setting (the 33 GB catalog fits neither a runner's disk nor GitHub's 10 GB cache) |
| `--only <id,id,...>` | all | sweep only these catalog `model_id`s (every id must be in the catalog) |
| `--skip-existing` | off | skip models whose `<out>/<model_id>.json` already exists — the resume flag |

Output match passes when every element satisfies
`|out - ref| <= max(tolerance_abs, tolerance_rel * |ref|)`. Latency is the
median wall time of `model.run` **plus output readback to CPU** — the
end-to-end cost a web app pays.

### Batches (what CI runs)

    npm run sweep:batches -- --catalog <csv> --out <dir> [--batch <n>] -- <harness flags...>

Runs the harness in a **fresh process per `--batch` models** (default 8),
because the runner accumulates memory across the models of one run (Traps,
below). A batch whose process dies is re-run one model at a time; a model that
kills the process even alone is printed as `UNMEASURED` and the driver exits
`1` — the output directory keeps every model that was measured, and a re-run
skips them. Flags after `--` go to every harness run. The weekly resweep uses
`--batch 8 -- --cache-dir /tmp/model-cache --cache-max-bytes 0` into `data/sweep-ci/`
(DECISIONS #166). Measured 2026-08-28 on `ubuntu-latest`: 157 models in 5 h 06 min, 159/159
with results, disk steady at 13 GB free.

Exit codes: `0` all result files schema-valid, `1` at least one failed
validation, `2` usage / manifest error. A model that fails to load, run, or
even crashes its page is a **result**, not an error — the batch continues
(fresh browser context per model × backend).

## Catalog manifest format (`data/web_catalog.csv` contract)

CSV with exactly these columns:

```csv
model_id,source,input_spec,license,notes
example-web-a,model_web_a_example.tflite,1x64:float32,Apache-2.0,tiny elementwise fixture
mobilenet-v2,https://huggingface.co/litert-community/x/resolve/main/m.tflite,1x224x224x3:float32,Apache-2.0,
```

- `model_id` — `^[a-z0-9][a-z0-9._-]*$`, unique; names the output file.
- `source` — `http(s)` URL or a path **relative to the manifest file**. Model
  artifacts are never committed to this repo. URL sources must be served with
  CORS headers (the harness page is cross-origin isolated; Hugging Face
  resolve URLs qualify).
- `input_spec` — either dimension specs `1x64:float32` (multiple inputs
  `;`-separated; dtypes `float32` | `int32` — LiteRT.js tensor I/O is
  float32/int32), or the path of an input-fixture JSON file
  (`{"inputs": [{"shape": [1,2], "dtype": "float32", "data": [..]}]}`),
  relative to the manifest. Dimension-spec inputs are generated
  deterministically from `--seed`.
- `license` — required per entry.
- `notes` — free text, echoed into the result file.
- Lines starting with `#` are full-line comments and parse as nothing — the
  catalog's exclusion blocks (the litertlm section, dynamic-dim exclusions)
  live in them. A `#` anywhere else in a line is literal cell content.

### Caveat: int32 mask inputs and quantized models (2026-08-13)

Two ways a record can read "numerically broken" when nothing on the GPU is:

1. **Dimension-spec int32 inputs are filled 0..15 — including attention masks.**
   Encoder-style graphs *multiply* the mask into activations, so a nonbinary
   fill inflates the activation range ~200×, which collapses the CPU (DRQ int8)
   reference while the GPU computes dequantized float. Measured on
   lfm2.5-encoder-230m wi8fc: max_abs_diff 197.7 with generated inputs vs 0.679
   (= the int8-vs-float noise floor) with a realistic fixture. For any model
   with a 0/1-semantics int32 input, use an input-fixture JSON with a binary
   mask instead of a dimension spec.
2. **DRQ-quantized models cannot pass the element tolerance against a float
   backend.** CPU runs int8 activation-quantized kernels; WebGPU dequantizes
   weights and computes float. `output_match: false` on a `*_wi8fc`/int4 model
   × `webgpu_mldrift` is expected even when both backends are healthy; judge
   those pairs at task level (or against a float export), not by 1e-5/1e-3.

Same catalog + same flags ⇒ identical output bytes **except latency values**
(`latency_p50_ms`), which are wall-clock measurements. Records are sorted by
backend id, JSON keys sorted, 2-space indent, trailing newline. Runtime
INFO-level log lines (which carry pointer addresses) are filtered out of
`delegation_evidence`; warnings and errors are kept verbatim.

## Backends

| Backend id | LiteRT.js accelerator | Notes |
|---|---|---|
| `wasm_xnnpack` | `wasm` | CPU reference for output match |
| `webgpu_mldrift` | `webgpu` | JSPI-enabled load; `full_delegation` from `CompiledModel.isFullyAccelerated`; Chromium launched with `--enable-unsafe-webgpu --use-angle=metal` so headless uses the real GPU (SwiftShader otherwise — check `env.webgpu_adapter`) |
| `webnn` | `webnn` | behind `--experimental`; public inclusion is an owner decision |

## Traps (measured)

**The runner accumulates memory across models in one run — batch size is the real limit, not artifact size.**
A 2026-08-13 delta sweep of 36 models died at 27 with
`FATAL ERROR: Ineffective mark-compacts near heap limit`, and re-running the remainder kept dying even after
excluding the two largest files and raising the heap to 12 GB. Sweeping the same models **one at a time**
landed all but two, including the 2.39 GB artifact that the batch runs had been blamed on. So: when a run
aborts, re-run the remainder in small batches (or singly) rather than hunting for a big file.

`--skip-existing` resumes a run (models with a result file in `--out` are skipped), and
`npm run sweep:batches` does the batching and retrying for you. A larger heap still helps:

    NODE_OPTIONS="--max-old-space-size=12288" npm run sweep -- --catalog <rest.csv> ...

**Some graphs OOM the runner on their own.** `qwen3-tts …__talker_fp32` and `…__talker_int4` both abort as
single-model runs at a 12 GB heap. The int4 is 0.26 GB, so this is the graph, not the bytes — record such
models as excluded in the catalog's `notes` instead of chasing them with more heap.

**A hosted CI runner cannot hold the catalog, and a harness that cannot launch its browser must still exit.**
The 2026-08-27 full-tier run (33032586477) pre-downloaded the catalog into `--cache-dir` — the catalog is 33 GB
(159 files, median 51 MB, 17 over 500 MB), the runner's disk filled after ~55 minutes (`ENOSPC`), Chromium then
failed to launch, and because the local server was still listening the process never exited: five silent hours
until GitHub's 6 h job cap. Nothing was measured. `actions/cache` could never have persisted that directory
either (10 GB limit — the memory notes from 2026-08-23 already said so). Hence: models are staged one at a time,
`--cache-max-bytes 0` in CI, the server and browser close in `finally`, and the CLI leaves within 10 s of the
run ending whatever else is alive.
