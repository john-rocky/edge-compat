# edge-compat

**Will this `.tflite` run on the GPU, the NPU, or in the browser — and where will it fall back?**
edge-compat answers that for [LiteRT](https://ai.google.dev/edge/litert) (Google's on-device
runtime, the new name for TensorFlow Lite) with measurements, not guesses: an op-level delegate
compatibility matrix (659 entries in 18 backend × runtime-version snapshots), `edge-lint` — a
static pre-flight that reads a `.tflite` and predicts delegate fallback before you run it — 203
model cards with 751 dated device runs (Galaxy S26, Pixel 8a, iPhone 17 Pro, Raspberry Pi 5,
Apple M4 Max), and a browser sweep of 92 models on `@litertjs/core` 2.5.3. Passing and failing
models are both listed. Counts as of 2026-09-08; every row carries its own date and environment.

Built by john-rocky. Measurements and views are my own.
edge-compat is an unofficial project for LiteRT, not an official Google product.

**Try it in two commands** — no clone, Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/):

```sh
curl -sO https://john-rocky.github.io/edge-compat/data/matrix/webgpu_mldrift__2.5.3.json
uvx --from git+https://github.com/john-rocky/edge-compat edge-lint model.tflite --matrix webgpu_mldrift__2.5.3.json
```

Pick the snapshot for your backend and runtime version from [`data/matrix/`](data/matrix/)
(one file per backend × version). An op with no measured entry reports `unknown` — never a
guess. To verify an install, run the shipped example after `uv sync` in a clone:
`uv run edge-lint data/examples/model_mixed_example.tflite --matrix data/examples/matrix_example.json`
(exit 0, "40.0% of 5 ops delegated · 3 blocking ops").

**Questions this repository answers**

- Which TFLite ops does the LiteRT GPU delegate (ML Drift, the CompiledModel GPU accelerator)
  run, fall back on, or compute wrong, per runtime version? → [`data/matrix/`](data/matrix/),
  `gpu_mldrift*__<version>.json`; every entry carries its evidence and provenance.
- Why does my model fall back to CPU, or return wrong numbers, on the GPU? → `edge-lint` names
  the op, the matched entry, and the rewrite the matrix recorded; `edge-fix` applies recorded
  transforms as a dry run.
- Which ops does the Qualcomm NPU path (QNN / HTP) accept or reject? →
  `data/matrix/npu_qnn_htp_android__*.json`.
- Does this model run in the browser with LiteRT.js on WebGPU, and how fast? → the
  [site table](https://john-rocky.github.io/edge-compat/) and [`data/sweep/`](data/sweep/).
- How fast is model X on device Y with which runtime version? → its card under
  [`cards/<model>/CARD.md`](cards/README.md), with the raw record under `data/device_runs/`.

*(Disclosure line and affiliation notice — single-sourced in `src/litert_compat/branding.py`,
mirrored in `site/src/shared/branding.ts`, rendered on every public surface and checked by
`edge-compat release-gate`.)*

**Public mirror.** The public repository is a scrubbed export of the project's private lab
repository, produced by `tools/export_public.sh`; `EXPORT.md` in the mirror names the source
commit. Numbered `DECISIONS`/`PROGRESS` citations in this README refer to the lab's logs.
Site: https://john-rocky.github.io/edge-compat/ — `llms.txt` at the site root.

Developer tooling built around one shared data asset: a **delegate compatibility matrix**
for LiteRT, derived from real conversions and real device measurements. LiteRT's GPU/NPU
delegates are closed-source and there is no public ground truth for op coverage; this repo's
matrix is that ground truth, exposed to humans, CI, and AI agents through stable JSON, exit
codes, and clean markdown.

**Machine consumption first.** All generated output is deterministic (sorted keys, stable
ordering, byte-identical reruns) so diffs are reviewable and pipelines can trigger on real
change.

## Not to be confused with

`litertlm-convert/qa/compat_check.py` (and `quant_scan.py`) check a **different axis**:
release-day *runtime-version × shipped-artifact loadability*. This repo's matrix is
**op-level delegate support** per (backend × LiteRT version). The two are complementary
and stay separate.

## Data integrity

- **No fabricated data, ever.** Every entry carries `provenance`:
  `measured | vendor_doc | inferred | example`. All real data comes from the owner's
  measurements; `data/examples/` is clearly marked `"provenance": "example"` and is what
  all tests run against.
- **One snapshot file per (backend × litert_version).** Delegate op support changes across
  runtime versions and differs per delegate, so backends and versions are never merged
  into one file, and every lookup verdict carries the snapshot's `backend` +
  `litert_version`.
- **Staleness is visible, never silent.** Snapshots are carried forward explicitly
  (`matrix carry-forward`), never edited in place: carried `measured` entries downgrade to
  `inferred` with `evidence.derived_from` pinning their source. The `inferred` count of a
  file *is* its staleness debt; `matrix validate` reports per-provenance counts.
- Op names are **exact TFLite builtin operator names** (`GATHER_ND`, `TRANSPOSE_CONV`, …).
  No aliases.

## Install

Python ≥ 3.11.

```sh
uv sync            # or: pip install -e .
uv run edge-compat --help
```

The Phase 8 backend runners (real probe measurement) and the `edge-fix`
numerical verification need the `runners` extra:

```sh
uv sync --extra runners    # or: pip install -e '.[runners]'
```

Without it every tool still works; runners just report "unavailable",
`release-check` keeps the affected entries honestly `inferred`, and
`edge-fix --apply` refuses unless `--allow-unverified` is passed.

## CLI

### `edge-compat matrix validate <file>`

JSON Schema validation (against `schemas/matrix.schema.json`) plus semantic lint:
contradictory duplicate entries for the same (op, dtypes, constraints) signature, missing
evidence on `measured` entries. Prints per-provenance entry counts.
Exit codes: `0` pass · `1` findings · `2` usage error.

### `edge-compat matrix import --csv <file> -o <matrix.json> --litert-version X --backend B [--device … --soc … --driver … --generated-at YYYY-MM-DD]`

Imports a curated CSV table (contract below). Forgiving on input — whitespace trimmed, any
column order, unknown columns warned and ignored. Strict on output — the result must pass
`matrix validate` or nothing is written. Deterministic: same CSV + same options →
byte-identical JSON (`--generated-at` defaults to today; pass it explicitly for
reproducible output).
Exit codes: `0` ok · `1` CSV data errors or invalid result · `2` usage error.

⚠ **Import REPLACES the output file — it does not merge.** Writing `-o` onto an existing
snapshot silently drops every entry not in the CSV (cost a recovery on 2026-08-14: a
1-row CSV clobbered the 3-row `gpu_metal_mac__0.16.0.json`; restored from git, merged by
hand, re-validated). To add rows to an existing snapshot: import to a temp file, append
the new entries into the existing JSON, then `matrix validate` — or keep the staging CSV
cumulative. Field vocab that trips first-time rows: `status` ∈ crash/delegated/fallback/
incorrect/partial/unknown (no "conditional"), dtypes are schema names (`float32`, not
"fp32"), `constraints` must parse as key=value.

### `edge-compat matrix diff <old> <new> [--json]`

Deterministic semantic change summary between two snapshots of the **same backend**:
entries added/removed, ops added/removed, `status` transitions, `provenance` transitions.
Free-text edits (conditions, rewrite hints) do not count as semantic change.
Exit codes: `0` no semantic difference · `1` differences found · `2` usage error
(including invalid input files and backend mismatch) — so a release watcher can trigger
on exit `1` knowing it always means real change.

### `edge-compat matrix carry-forward <file> --to-litert-version X.Y.Z -o <new> [--generated-at YYYY-MM-DD]`

Produces the next-version snapshot; refuses to write in place. Every `measured` entry is
downgraded to `provenance: "inferred"` with `evidence.derived_from` recording the source
snapshot's `litert_version` and `generated_at`. Re-measurement later upgrades entries back
to `measured`.
Exit codes: `0` ok · `2` usage error.

### `edge-compat probe gen [--from-lint report.json]… [--from-matrix snapshot.json]… [-o data/probes] [--json]`

Builds minimal single-op `.tflite` probe fixtures from complete lookup signatures —
`needs_probe` records of `edge-lint --json` reports, and/or the `inferred` entries of a
matrix snapshot. Deterministic: signature → stable fixture id
(`<OP>__<dtypes>__<meta-hash>`) and byte-identical file. Signatures the template registry
cannot realize are listed as `unprobeable` with a reason — never approximated (see
"Release pipeline" below).
Exit codes: `0` all generated · `1` unprobeable signatures present · `2` usage error.

### `edge-compat release-check <snapshot.json>… --to-litert-version X.Y.Z -o <out-dir> [options]`

The one command per runtime release: carry-forward → probe → re-measure → upgrade → diff
→ report. See "Release pipeline (Phase 8)" below.
Exit codes: `0` nothing changed · `1` semantic changes found · `2` error.

### `edge-compat release-gate [--repo-root .] [--scope data/release_scope.json] [--site-dist site/dist]`

The pre-publication gate (Phase 12): run by the release runbook and
the `deploy-zoo` CI job before anything goes public. Deterministic; three guards:

1. **Benchmark scope** — no card presents a non-example `cross_runtime` comparison for a
   model outside the approved public scope in `data/release_scope.json` (owner-editable
   data; initially Gemma 4 only per the launch plan — widening it is a data change, not a
   code change).
2. **Disclosure** — the disclosure line (`src/litert_compat/branding.py`, mirrored in
   `site/src/shared/branding.ts` with test-enforced parity) renders in `README.md`,
   `cards/README.md`, `llms.txt`, and — with `--site-dist` — every generated HTML page.
3. **Example banners** — cards carrying `example`-provenance records must render their
   example banner; the site index must show its fixtures banner when example records are
   displayed.

Exit codes: `0` pass · `1` findings · `2` usage error.

## `edge-lint` — static delegate pre-flight

```sh
edge-lint MODEL.tflite \
  --matrix data/matrix/<backend>__<litert_version>.json \
  [--backend gpu_mldrift] \
  [--json | --md] \
  [--fail-on fallback] [--fail-on partitions:N] \
  [--rules data/transforms]
```

Statically lints a `.tflite` model against a matrix snapshot — **no LiteRT
runtime, no model execution**: the flatbuffer is parsed directly, so the linter
runs anywhere Python runs. LiteRT-LM `.litertlm` bundles are accepted too
(detected by magic bytes): the embedded TFLite model is extracted from the
container TOC and linted statically — the report's `model.path` is marked
`::TFLiteModel@section<i>` and `model.sha256` hashes the analyzed embedded
bytes, so container analysis is always visible as such. Exactly one
`TFLiteModel` section is required; ambiguous multi-model bundles are refused
with the section listing (E2E bundle audits stay in litertlm-convert).
Pipeline: parse → classify every node through
`Matrix.lookup` (signature from the node's output tensors) → simulate
partitions (maximal runs of delegate-claimed nodes in execution order; every
partition boundary implies a host↔device sync) → rank blocking ops by a
labeled heuristic → report.

- **Formats**: human text (default), `--md` markdown, `--json` validating
  against `schemas/lint_report.schema.json` (a public, versioned API).
- **Claimed statuses**: `delegated`, `partial`, and `incorrect` count as
  claimed by the delegate for partition simulation — `incorrect` ops *do* run
  on the delegate, just wrongly, and are additionally surfaced as correctness
  warnings. `unknown` never claims: no matrix entry, no guess.
- **Rewrite suggestions** are surfaced verbatim from matched entries'
  `rewrite_hints` — never generated.
- **Needs-probe list**: every unknown-verdict builtin op is emitted as a
  complete lookup signature (op, dtypes, shape_meta, backend) — sufficient to
  construct a probe without reopening the model (consumed by the Phase 8 probe
  suite). Custom ops are listed separately; they are outside the matrix
  vocabulary and not probeable.
- **`--backend`** is optional and defaults to the matrix file's backend; a
  mismatch is a usage error — one snapshot per (backend × litert_version),
  and no output ever presents one backend's result as another's.

Exit codes: `0` clean · `1` a `--fail-on` rule triggered · `2` usage/parse
error. `--fail-on fallback` triggers on `fallback`, `incorrect`, **and**
`crash` verdicts; `--fail-on partitions:N` triggers when the partition count
exceeds N; the flag is repeatable and rules combine.

### Static pre-flight vs runtime confirmation

The upstream `litert_gpu_toolkit` checker
([`checker.py` in google-ai-edge/litert-samples](https://github.com/google-ai-edge/litert-samples),
`check_gpu_compatibility()`) is a **runtime compile-and-compare** check: it
needs TensorFlow and the LiteRT runtime, actually compiles the model for GPU,
and returns a binary verdict. `edge-lint` is its **static complement**:
flatbuffer-only, matrix-backed, partition simulation, stable JSON and exit
codes, and it names *which* op blocks and *why* (with the matrix entry's
provenance). Run both: `edge-lint` as the cheap pre-flight everywhere (CI,
agents, laptops), the runtime checker as confirmation where a runtime and
device are available.

### CI example (GitHub Actions)

```yaml
- name: LiteRT delegate pre-flight
  run: |
    pip install git+https://github.com/john-rocky/edge-compat
    edge-lint model.tflite \
      --matrix data/matrix/gpu_mldrift__2.2.0.json \
      --fail-on fallback --fail-on partitions:1 \
      --json > lint_report.json
  # exit 1 fails the job when any op falls back / crashes / is numerically
  # incorrect, or when the graph splits into more than one partition
```

## `edge-card` — perf cards

Every published model gets a card: **`card.json` is the machine-readable source
of truth** (validates against `schemas/card.schema.json`), `CARD.md` is the
rendered view — clean, standalone markdown (no HTML, no relative asset links)
for GitHub, RAG systems, and AI crawlers. This phase formats and links measured
data; **it never produces measurements** — every benchmark record carries
`provenance`.

`CARD.md` is **not** a replacement for the hand-written Hugging Face model
cards: `card.json` is the structured superset those cards' Performance tables
are derivable from (first-class fields for the frozen litert-community
perf-table columns, the open `metrics` map for harness-specific ones).

### `edge-card build`

```sh
edge-card build \
  --model model.tflite --meta meta.yaml \
  [--bench results.json] [--lint lint.json] [--cross-bench results.json] \
  -o cards/<model_id>/
```

- `--bench` / `--cross-bench`: benchmark results files
  (`schemas/benchmark_result.schema.json`; contract below). Cross-runtime
  records additionally require a `runtime` field.
- `--lint`: a `edge-lint --json` report; the card's `delegation` block is
  mapped directly from it and always carries the matrix snapshot's `backend` +
  `litert_version`. The report must describe the exact `--model` file (sha256
  is checked), and bench files must name the card's `model.id` — identity
  mismatches are refused.
- Exit codes: `0` written · `1` invalid input data (nothing written) · `2`
  usage error.

### `meta.yaml` — the human-authored fields

```yaml
model:
  id: example-tiny-clean            # catalog id; also the card directory name
  family: example-family            # model family, e.g. "mobilenet", "olmo2"
  task: image-classification        # free-text task label
  source_url: https://example.invalid/models/example-tiny-clean
  license: apache-2.0

conversion:
  tool: example-converter
  tool_version: 0.0.0-example
  command: example-converter --input model.onnx --output model.tflite
  quantization: none                # e.g. "none", "int8-dynamic", "int4-block32"

pitfalls:                           # optional; surfaced verbatim — never generated
  - Free-text pitfall the owner wants next to the numbers.
```

Unknown keys are warned and ignored; missing required fields are collected and
reported together.

### Benchmark results file (`schemas/benchmark_result.schema.json`)

```json
{
  "schema_version": "1.0",
  "model_id": "example-tiny-clean",
  "results": [
    {
      "device": "…", "soc": "…", "backend": "gpu_mldrift", "precision": "f32",
      "latency_p50_ms": 1.2, "tokens_per_s": null, "peak_mem_mb": 12.5,
      "metrics": {"encode_ms": 0.5, "decode_ms": 0.7, "rtf": 0.01},
      "harness": "…", "harness_version": "…", "runs": 10, "statistic": "median",
      "date": "2026-08-10", "provenance": "measured"
    }
  ]
}
```

Required per record: `device`, `backend`, `harness`, `date`, `provenance`.
`metrics` is the open map for harness-specific values (stage latencies, `rtf`,
`prefill_tok_s`, `ttft_s`, `power_w`, …) so existing exports — devicemark
leaderboard JSONL, bench-harness exports, gpu_audit outputs — map on via
adapters (`src/litert_compat/cards/adapters/`; stubs until the owner supplies
sample exports — the mapping is never guessed).

### `edge-card build-all`

```sh
edge-card build-all --manifest manifest.csv [--out cards/]
```

Manifest CSV columns (`model`, `meta` required; the rest optional per row;
paths resolve **relative to the manifest file**):

```csv
model,meta,bench,lint,cross
model_a.tflite,meta_a.yaml,bench_a.json,lint_a.json,cross_a.json
model_b.tflite,meta_b.yaml,,lint_b.json,
```

Each card is written to `<out>/<model.id>/`. Failures are **collected and
reported together at the end** — never fail-fast; exit `1` if any row failed.
The example manifest (`data/examples/cards_manifest_example.csv`) includes a
deliberately broken row to exercise exactly that path.

### `edge-card index`

```sh
edge-card index cards/
```

Deterministically generates `cards/index.json` (machine-readable catalog,
including per-model browser statuses), `cards/README.md` (one-line-per-model
table with a Browser column), and `llms.txt` at the repo root — the single
entry point linking every `CARD.md`, the matrix snapshots under `data/matrix/`,
and the browser sweep records behind enriched cards, from which AI agents and
crawlers can discover the whole dataset. Invalid cards abort with exit `1` and
nothing written. The committed `cards/` content and `llms.txt` are byte-pinned
to tool output by tests.

### `edge-card enrich`

```sh
edge-card enrich --sweep data/sweep/ [cards/]
edge-card enrich --device-runs-root data/device_runs [cards/]   # device lane, see Phase 13
```

Merges browser sweep results (Phase 5 harness output, one
`sweep_result.schema.json` file per model) into the matching cards: each card
gains a `browser` block — fields mapped **1:1 from the sweep record** (loads /
runs / full delegation / output match / max rel diff / latency, plus the full
`env` block and `provenance`), with `sweep_source` linking the raw sweep JSON
and a nullable `demo_url` (filled in Phase 7; preserved across re-enrichment).
Enriched cards declare card schema `1.1`; `CARD.md` gains a
"Browser (LiteRT.js)" section with env-labeled latency. The catalog outputs
(`index.json`, `cards/README.md`, `llms.txt`) are regenerated through the same
code path as `edge-card index`.

- Sweep results without a matching card are **noted and skipped** (a sweep may
  cover models before their cards exist); cards without a sweep result are left
  byte-for-byte untouched.
- **Zero matrix writes, ever** (the trap rule — see "Browser backends" below);
  covered by a test.
- Exit codes: `0` written · `1` invalid sweep result or card, or a device cell
  a card carries would be dropped without `--allow-drop` (nothing written)
  · `2` usage error.

Index browser-status vocabulary (a summary cell — the evidence stays in the
card and the linked sweep record): `load_failed` / `run_failed` /
`output_mismatch` (ran, numerics outside tolerance vs the wasm_xnnpack
reference) / `fallback` (ran, but not fully delegated) / `pass` (loaded, ran,
and neither a mismatch nor a fallback was recorded).

## `web/sweep` — LiteRT.js browser sweep harness

The Phase 5 instrument (TypeScript + Playwright, the repo's one non-Python
lane): batch-measures real `.tflite` models on LiteRT.js in headless Chromium
— loads / runs / full-delegation / output-match-vs-CPU / latency p50, per
model × backend (`wasm_xnnpack`, `webgpu_mldrift`, `webnn` behind
`--experimental`) — writing one
[`schemas/sweep_result.schema.json`](schemas/sweep_result.schema.json)-valid
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
determinism guarantee are documented in [web/sweep/README.md](web/sweep/README.md).
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
weights guard). Details: [site/README.md](site/README.md). The site is deployed
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

## Release pipeline (Phase 8) — probe suite & `release-check`

A runtime release used to mean a pile of manual re-checks. The matrix's anti-staleness
contracts (`carry-forward` makes staleness visible, `diff` makes change diffable, lint
emits probe-ready `needs_probe` signatures) reduce it to one command:

```sh
edge-compat release-check data/matrix/gpu_mldrift__1.2.0.json \
  --to-litert-version 1.3.0 \
  --lint-report reports/model_a.lint.json \
  --manifest manifest.csv \
  -o release/1.3.0/
```

Pipeline, in order: carry each snapshot forward (measured → inferred) → generate probe
fixtures for the `inferred` entries plus accumulated `needs_probe` signatures → run every
**available** runner → upgrade re-measured entries back to `measured` → `matrix diff` old
vs new → emit a deterministic release report (`release_report.md` + `.json`) covering:
what changed, what remains `inferred` (the staleness debt), what needs a device attached,
and — given a Phase 3 `build-all` manifest — which models' lint verdicts changed.
Headless-safe: an unavailable backend is remaining work, not an error.

**Release-watcher hook** (the owner's existing watcher detects new litert versions; wire
its "on new version" action to):

```sh
edge-compat release-check data/matrix/*.json --to-litert-version "$NEW" -o "release/$NEW/" \
  || echo "semantic changes (or error) — review release/$NEW/release_report.md"
```

Exit `0` means the new runtime measures identically; exit `1` means real semantic change
(including staleness that could not be re-measured); exit `2` means error.

### Probe fixtures (micrograph honesty)

`probe gen` builds single-op fixtures that the **real** runtime executes (constant
tensors carry real buffers; ops carry real builtin options). Every fixture is round-trip
verified: parsed back and re-classified to prove it exercises exactly the requested
(op, dtypes, shape_meta). A probe measures the op **in isolation** — full-graph behavior
can differ (fusion, memory pressure) — so probe-derived entries record
`evidence.source_model: "probe:<fixture-id>"`, keeping them distinguishable from
full-model measurements, and a probe result that contradicts a full-model `measured`
entry is surfaced as a **conflict** in the report, never silently overwritten in either
direction.

Template registry: option-free unary elementwise ops, binary elementwise ops
(ADD/SUB/MUL/DIV/…), SOFTMAX (beta=1.0), static RESHAPE — plus the graduated
weighted/structured fp32-activation families (DECISIONS #147–#149): CONV_2D,
DEPTHWISE_CONV_2D, TRANSPOSE_CONV, FULLY_CONNECTED, the pools, BATCH_MATMUL,
CONCATENATION, TRANSPOSE, SLICE, PAD, SUM/MEAN/REDUCE_MAX/REDUCE_MIN,
LEAKY_RELU, CAST, GREATER/EQUAL, SELECT/SELECT_V2, the resizes, and
EMBEDDING_LOOKUP. Constant weights are signature-seeded (same signature →
byte-identical fixture); each fixture is one fixed minimal instantiation
(SAME padding, stride 1, 2×2 kernels, depth 4) with honest op versions.
Anything else — quantized-weight forms (DEQUANTIZE included), dynamic-shape
weighted signatures, option-constrained forms — is reported `unprobeable`
with a precise reason (extending the registry is additive; see PROGRESS.md
"Deferred").

### Runners

| runner | claims backend | needs |
|---|---|---|
| `cpu` | `cpu`, `cpu_xnnpack` | the `runners` extra (ai-edge-litert interpreter) — the numeric reference |
| `webgpu_mac` | `webgpu_mac` | macOS + the `runners` extra (CompiledModel GPU) |
| `gpu_mldrift_adb` | `gpu_mldrift` | `adb`, a connected device, and an on-device helper (below) |
| `npu_adb` | `npu_qnn`, `npu_neuropilot` | `adb`, a connected device, and an NPU-capable on-device helper (Phase 13; below) |
| `webgpu_playwright` | `wasm_xnnpack`, `webgpu_mldrift` | the `runners` extra (CPU reference) + Node ≥ 20 with `web/sweep` installed (`npm ci && npm run build`) |

Rules: correctness is part of status — compiles-and-runs but outside the numeric
tolerance is `incorrect`, not `delegated`. A runner never guesses: no result, no entry.
Inputs are seeded and deterministic (mulberry32, values in `[0.25, 1.25)` so
domain-restricted ops stay NaN-free). Default tolerance vs the CPU reference is
`|out − ref| ≤ max(1e-5, 1e-3·|ref|)` (override with `--tolerance-abs` /
`--tolerance-rel`; per-backend policy is an open owner decision).

The `webgpu_mac` runner records which delegate actually served by scanning the runtime
log (on ai-edge-litert 2.1.6 macOS wheels, `HardwareAccelerator.GPU` is served by the
**Metal** delegate) — the result note carries that evidence, so a Mac host-GPU number can
never masquerade as another backend's.

The adb runners need an owner-provided on-device helper (defaults
`/data/local/tmp/litert-probe-runner` for `gpu_mldrift_adb` and
`/data/local/tmp/litert-npu-probe-runner` for `npu_adb`; override with
`$LITERT_COMPAT_ADB_HELPER` / `$LITERT_COMPAT_NPU_ADB_HELPER`) with this contract:
invoked as `<helper> --model <pushed.tflite> --backend <backend>`, it prints one JSON
line `{"status": <matrix status enum>, "max_abs_diff": <float|null>, "note":
<optional string>}` and exits 0. Missing adb/device/helper → the runner reports
"unavailable" and affected entries stay `inferred`. NPU probes may additionally
require per-SoC AOT compilation (vendor SDK linkage is the owner's helper build);
signatures the helper cannot compile surface as unprobeable-with-reason, never as a
fabricated status.

### The web runner (`webgpu_playwright`) — op-level web matrix entries

The graduated deferred item: browser probes through the Phase 5 sweep harness,
the designed path that lifts the Phase 6 trap rule with real measurements
instead of model-level inference. Per probe, the runner serves the fixture and
the harness page from the sweep's local server, drives headless Chromium
(WebGPU flags, JSPI), and maps the outcome onto the matrix status enum.

- **Input injection, byte-identical:** the exact seeded feeds used for the CPU
  reference are materialized into a job file and fed to the page verbatim
  (`web/sweep/src/node/probe.ts`; the page never generates probe inputs).
  Outputs cross back verbatim; the numeric comparison runs in Python against
  the same ai-edge-litert CPU reference with the shared tolerance. Dtypes
  outside LiteRT.js tensor I/O (float32/int32) are refused precisely, never
  coerced.
- **Status mapping:** webgpu — compile refusal → `fallback`, partial
  acceleration → `fallback`, run failure → `crash`, numerics decide
  `delegated` vs `incorrect`. `wasm_xnnpack` is the CPU-class backend (nothing
  to fall back to): load/run failures are `crash`.
- **Version axis (important):** snapshots this runner populates live on the
  `@litertjs/core` version axis — pass e.g. `--to-litert-version 2.5.3` to
  `release-check`, matching the installed core package. Every result note
  records the measured core version plus the WebGPU adapter as evidence, so a
  mismatch is visible in the release report.
- Unavailable without Node, the web/sweep install, or the harness bundle — the
  release report says exactly which, and affected entries stay `inferred`.

## `edge-fix` — diagnosis → repair (Phase 10)

The matrix's `rewrite_hints` are human prose; `edge-fix` promotes them to
executable, deterministic graph surgery: lint finds why a model won't delegate, fix
makes it delegate — then proves it didn't change the math.

```sh
edge-fix MODEL.tflite --backend gpu_mldrift \
  --matrix data/matrix/<snapshot>.json --rules data/transforms/ \
  [--dry-run (default) | --apply --out FIXED.tflite] \
  [--json report.json] [--allow-unverified] \
  [--tolerance-abs 1e-5] [--tolerance-rel 1e-3]
```

Pipeline: **lint** the model → **match** transform rules to the non-delegated
findings → **plan** (ordered, conflict-checked: a node claimed by two rules is left
untouched and reported) → **apply** on a loaded copy via the flatbuffer object API →
**re-lint**: the metric must improve — delegated coverage first, then
incorrect+crash count, then partition count; `io_cast` plans may instead reduce the
interface-violation count without regressing the lint metrics — else automatic
rollback and a `no_improvement` report → **numerical verification**: original vs
fixed on the LiteRT CPU interpreter (the same `edge-compat[runners]` extra and
tolerance policy as the sweep and probe runners) on seeded deterministic inputs.

Safety defaults, all tested: dry-run is the default; the input file is never
modified in place; a `schemas/fix_report.schema.json`-valid report is always
emitted; a **failed** verification rolls back and is never overridable. Without the
runners extra, `--apply` refuses unless `--allow-unverified` is explicitly passed —
and then the report is marked `unverified`. Deterministic and idempotent: same
inputs → byte-identical fixed model and report; running fix on its own output
applies zero further changes.

Exit codes: `0` fixed (or nothing to fix) · `1` unfixed issues remain or automatic
rollback · `2` usage/data error (including the `--apply` verification refusal and
models the rewrite path cannot round-trip faithfully — multiple subgraphs, custom
ops, custom quantization `details`, unregistered builtin-options tables are refused
with precise reasons, never silently dropped). Real converter output — quantized
tensors, model metadata, signature defs, appended-weight buffers, optional `-1`
inputs — round-trips faithfully (DECISIONS #153–#155).

### Transform rules: machinery from the toolkit, knowledge from the owner

Rules are JSON files (`<id>.json`, `schemas/transform_rule.schema.json`) in a rules
directory — `data/transforms/` for the owner's real rules (none yet; the committed
example-provenance rules live in `data/examples/transforms/` and are exercised only
on synthetic fixtures). Every applied change in a fix report traces to a rule `id`
with its evidence; unmatched findings are reported, never guessed at. Matrix schema
`1.2` adds the link in the other direction: a `rewrite_hints` entry may carry a
`transform_id` naming the rule that executes it (surfaced verbatim by `edge-lint`,
report schema `1.1`). With `edge-lint --rules DIR`, hints whose `transform_id`
resolves to a rule in DIR are annotated **fixable by edge-fix** in the
text/markdown formats, with the exact `edge-fix` command; a dangling
`transform_id` is called out as unresolvable. The JSON report never changes —
it carries `transform_id` verbatim for machine consumers to resolve themselves.

A rule: `{id, title, match: {op | op_sequence, dtypes, constraints}, action,
params, expected_effect, provenance, evidence, notes, fixture}`. `match` uses the
exact `Matrix.lookup` semantics (dtypes subset, constraints vs shape_meta).
`fixture` declares the rule's own demonstration model + matrix; `edge-fix rules
validate` dry-runs the rule against it and requires the fix to succeed end-to-end.
(`op_sequence` is part of the schema API but not executed by the v1 engine — such
rules are refused with a precise reason, not errors.)

`action` is a **closed vocabulary** — anything not expressible in it is not a rule;
the vocabulary is extended by spec amendment, not freeform actions. One worked
example per action (each committed under `data/examples/transforms/`):

- **`replace_op`** — swap the operator in place, inputs/outputs verbatim.
  `"params": {"new_op": "FLOOR_DIV"}` — `example-replace-div-floor-div` rewrites
  integer `DIV` → `FLOOR_DIV` (exact for non-negative operands; the validity caveat
  lives in the rule's `notes` and is surfaced in every report).
- **`decompose`** — replace the node with a fixed recipe of nodes.
  `"params": {"nodes": [{"op": "MUL", "inputs": ["in:0", "in:0"], "outputs":
  ["out:0"]}]}` — `example-decompose-square-mul` rewrites `SQUARE(x)` → `MUL(x, x)`.
  Refs: `in:K`/`out:K` are the matched node's tensors, `tmp:NAME` a declared
  intermediate whose dtype/shape borrow from a matched tensor (`like:in:K`).
- **`insert_cast`** — compute the node in another dtype, CASTs at the edges.
  `"params": {"to": "float32"}` — `example-insert-cast-mul-int32` computes integer
  `MUL` in float32 (exact while products stay under 2^24).
- **`io_cast`** — wrap graph I/O to satisfy interface constraints (e.g. the
  LiteRT.js float32/int32 tensor I/O rule); the interior graph is unchanged.
  `"match": {"dtypes": ["int64"]}, "params": {"to": "int32", "io": "both"}` —
  `example-io-cast-int64`.

Rule-emitted ops are limited to the v1 options registry (option-free unary ops,
binary elementwise ops, `CAST`, `RESHAPE`) — a rule cannot emit an op whose
builtin-options contract is unknown. Extending the registry is additive, the same
pattern as the probe templates.

### Authoring path

```sh
edge-fix rules new --id my-rule --action replace_op --op DIV \
  --out-dir data/transforms/          # scaffold with TODO markers
# fill the TODOs, then:
edge-fix rules validate data/transforms/my-rule.json
# PASS requires: schema + semantic checks green, and the declared fixture
# dry-run reaches outcome `fixed` (metric improves; verification passes when
# the runners extra is installed).
```

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

## Device-run results (Phase 13) — the NPU 一覧 and the LLM 対応表

Matrix schema `1.3` registers the NPU backend IDs — `npu_qnn` (Qualcomm Hexagon
via QNN), `npu_neuropilot` (MediaTek) — with **zero entries** (the Phase 6
pattern; final ID set pending the owner's device fleet). NPU behavior varies
per SoC and vendor SDK, so `target` (`device`, `soc`, `driver` — the driver
field carries the vendor SDK version) is **mandatory on NPU snapshots**,
enforced by `matrix validate`. `edge-lint --backend npu_qnn` runs fine and
honestly reports `unknown` until log-backed or probe-backed entries exist.

Model-level facts live in **device-run results**
(`schemas/device_run_result.schema.json`, deliberately parallel to the sweep
schema): per (model × device × accelerator) — loads / runs / delegation
signals (with delegated-op counts when the runtime log states them) /
output-match vs the CPU reference / CV latency / LLM metrics
(`prefill_tokens_per_s`, `decode_tokens_per_s`, `ttft_ms`, `peak_mem_mb`,
context length) — each with verbatim log evidence and a mandatory `env` block
(`runtime` is `litert` for the .tflite-on-NPU lane or `litert-lm` for the
.litertlm lane; that choice also selects the staleness axis in
`data/releases.json`, where Phase 13 added the `litertlm` axis watched by
autobump). Snapshots are append-only:
`data/device_runs/<runtime_version>/<date>/<model_id>__<device>.json`, one
runtime per snapshot directory, verified at discovery.

**The trap rule, device edition:** op-level matrix entries are **never derived
from model-level device-run results** — in either lane. Op-level entries
require a runtime log naming the specific op (captured as `evidence` — the
gpu_audit logs contain exactly such extractions) or an `npu_adb` probe result.
Enforced by the same zero-matrix-writes tests as the web lane.

Ingestion adapters wrap the owner's EXISTING measurement instruments — those
tools remain the instruments; this repo is the structured home of their
results. Mappings were built against real staged samples
(`data/examples/llm_samples/`, owner decisions recorded in its README):

```sh
# gpu_gate_mac.sh logs (LiteRT-LM Mac GPU gate). Verdict comes from the LOG
# BODY only — the CLI exits 0 even on failure, and a results block with
# 0.00 tokens/s is a zero_throughput FAILURE, not a measurement. Fields the
# log does not carry (runtime version, date, host) are owner-supplied options;
# --speeds-contaminated ingests a retracted batch's verdict while withholding
# every speed figure.
edge-compat device-run ingest-gpu-audit model.gpu.log \
  --model-id gemma3-1b --device mac-m4-max --device-name "M4 Max" \
  --runtime-version 0.15.0 --date 2026-07-23 --out data/device_runs

# devicemark leaderboard rows: only *__litertlm rows are this lane (coreai
# rows belong in a card's cross_runtime list and are skipped with a note);
# mem_measured:false memory is carried as metrics.peak_mem_est_mb, never as a
# measured peak_mem_mb.
edge-compat device-run ingest-devicemark measurements.jsonl \
  --runtime-version 0.15.0 --date 2026-07-22 --accelerator cpu

# compat_check reports (litert-lm version × artifact loadability; this axis is
# NOT absorbed into the matrix). Refuse-until-sampled: status branches with no
# staged real sample (BROKEN, SUSPECT) are refused by name, never invented.
edge-compat device-run ingest-compat-check compat_0.15.0.json \
  --date 2026-08-10 --device mac-m4-max --accelerator cpu

edge-compat device-run validate data/device_runs   # schema + layout, exit 0/1/2
edge-compat device-run validate data/device_runs --cards cards   # + every card is the
                                                     # newest measurement per cell (#165)
```

`edge-card enrich --device-runs-root data/device_runs` merges records into
cards as the additive schema-1.2 `device` block (accelerator records embedded
verbatim): every snapshot under the root is read and each (device x
accelerator x prompt-length) cell takes its newest measurement, so no snapshot
is hand-picked (DECISIONS #165). `--device-runs <snapshot-dir>` (repeatable)
enriches from exactly those snapshots instead — and because a card's device
block is rebuilt from what is passed, a run that would drop a cell a card
carries today is refused with nothing written unless `--allow-drop` states the
drop. Enrichment adds the Device column to `cards/index.json` /
`cards/README.md`, links snapshots from `llms.txt`, and the site index gains
env-labeled, stale-flagged device columns. `device-run validate --cards`
checks that the cards are exactly that selection; `tests/test_device_coverage.py`
runs the same check against the committed tree. `edge-compat freshness
delta` diffs the two most recent device-run snapshots **per runtime** into
`DELTA_LOG.md` alongside the sweep entries.

## Query API

```python
from pathlib import Path
from litert_compat.matrix import Matrix

matrix = Matrix.load(Path("data/examples/matrix_example.json"))
verdict = matrix.lookup(
    "FULLY_CONNECTED",
    dtypes=["int8"],
    shape_meta={"rank": 2, "dynamic_shape": False},
)
# Verdict(status=..., matched_entry=..., reason=...,
#         backend="gpu_mldrift", litert_version="0.0.0-example")
```

Precedence, most-specific match wins: **dtypes AND constraints** beat **dtype-only** beat
**constraints-only** beat **op-only**. No match returns `status: "unknown"` — never a
guess, never a default to `delegated`. Constraint keys `max_rank`/`min_rank` compare
numerically against `shape_meta["rank"]`; other keys match by equality; missing
`shape_meta` information never satisfies a constraint. Ties at equal specificity with
contradictory statuses raise an error and are reported by `matrix validate`.

## CSV column contract

Column names map 1:1 to schema fields. Column order is free; header names are
case-insensitive; whitespace is trimmed. Multi-value cells use `;` separators. One rewrite
hint per row.

| Column | Maps to | Format |
|---|---|---|
| `op` | `entry.op` | required; exact TFLite builtin name, e.g. `FULLY_CONNECTED` |
| `dtypes` | `entry.dtypes` | `;`-separated, e.g. `float32;int8`; empty = op-generic entry |
| `constraints` | `entry.constraints` | `;`-separated `key=value`; values parsed as JSON scalars (`true`, `4`, `1.5`), else string |
| `status` | `entry.status` | required; `delegated \| partial \| fallback \| incorrect \| crash \| unknown` |
| `conditions` | `entry.conditions` | free text (quote the cell if it contains commas) |
| `rewrite_symptom` | `rewrite_hints[0].symptom` | free text; required if `rewrite` is set |
| `rewrite` | `rewrite_hints[0].rewrite` | free text; required if `rewrite_symptom` is set |
| `rewrite_expected_effect` | `rewrite_hints[0].expected_effect` | free text, optional |
| `provenance` | `entry.provenance` | required; `measured \| vendor_doc \| inferred \| example` |
| `evidence_source_model` | `evidence.source_model` | model id |
| `evidence_litert_version` | `evidence.litert_version` | version string |
| `evidence_date` | `evidence.date` | `YYYY-MM-DD` |

Snapshot-level fields (`litert_version`, `backend`, `target.*`, `generated_at`) are CLI
options, not CSV columns — one CSV import produces exactly one (backend × version)
snapshot.

`status: incorrect` means the delegate compiles and executes the op but produces numerics
outside tolerance vs the CPU reference — an observed, real failure mode and the worst one
for users.

## Freshness system (Phase 11)

The base rules keep staleness *visible* (every record dated, versioned,
env-labeled) and Phase 8 automates native re-measurement. The freshness
system keeps the **browser lane** current where automation honestly can, and
turns what it cannot automate into scheduled reminders instead of silent
decay. Hard rule, enforced by test: freshness automation performs **zero
writes under `data/matrix/`** — matrix truth is written only by Phase 8's
`release-check` probe pipeline (native) and, for web backends, by the
still-deferred browser probe runner. The Phase 6 trap rule is never
overridden.

### `data/releases.json` — the latest-known-release registry

The reference point for every staleness check: the newest *known* upstream
release per version axis (`litert` = ai-edge-litert on PyPI, `litertjs_core`
= @litertjs/core on npm). Maintained by `autobump.yml` or the owner — never
by measurement tools. It changes no verdict; it only decides whether a
verified-against version is flagged stale.

### Staleness surfacing (spec 11.3)

- `edge-lint` emits a **soft warning on stderr** when the matrix snapshot's
  `litert_version` lags the registry (`--releases <file>`; defaults to
  `data/releases.json` when present). Warning only — report bytes, verdicts,
  and exit codes are unchanged, proven by test. Placeholder versions
  (`0.0.0-example`) never warn.
- The demo-zoo index, demo pages, and the playground's op cross-reference all
  display *verified-against version + date*, with a visible `stale` flag when
  that version lags the registry (site generator `--releases` flag, same
  default). Display only; the flag never changes a status or measurement.

### `edge-compat freshness` commands

```sh
# Diff the two most recent sweep snapshots under data/sweep/<version>/<date>/
# — and, per runtime, the two most recent device-run snapshots under
# data/device_runs/ (Phase 13) — and upsert the generated entries into
# DELTA_LOG.md (never hand-edited).
# Exit: 0 no observed changes · 1 changes recorded · 2 usage error.
edge-compat freshness delta [--sweep-dir data/sweep] [--log DELTA_LOG.md] \
  [--latency-threshold-pct 25] [--rules-dir data/transforms] [--json]

# List manual-verify register items whose re-verify interval has lapsed.
# Exit: 0 nothing overdue · 1 overdue items · 2 usage error.
edge-compat freshness overdue [--register MAINTENANCE.md] [--as-of YYYY-MM-DD] [--json]
```

Sweep snapshots are **append-only**: one full sweep = one
`data/sweep/<litertjs_version>/<YYYY-MM-DD>/` directory of
`<model_id>.json` result files; history is what makes deltas computable.
`DELTA_LOG.md` is the web-lane counterpart of Phase 8's release report:
per-model/per-backend regressions and improvements (index status
vocabulary), latency shifts beyond the threshold, catalog changes, and
transform rules currently flagged stale.

### `edge-fix rules reverify` (spec 11.4)

```sh
edge-fix rules reverify data/transforms [--runtime-version X] [--date YYYY-MM-DD] [--json]
# Exit: 0 all pass · 1 stale rules · 2 data error / verification unavailable
```

Re-runs every rule's declared fixture against the current runtime. A
**measured** failure flags the rule `stale` in its file (transform_rule
schema 1.1, additive) — never auto-deleted; a later passing run removes the
flag. Without the `runners` extra nothing can be measured, so nothing is
ever flagged (exit 2 instead). Stale rules are listed in the next
`DELTA_LOG.md` entry.

### The four workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `resweep.yml` | weekly · `@litertjs/core` release (via autobump) · manual | quick tier (harness self-test + example catalog) every trigger; full catalog weekly/on release. Commits a new dated snapshot under `data/sweep-ci/` (a root separate from the owner's Mac snapshots in `data/sweep/`; the Mac stays the catalog's sweep axis and CI is a labelled second lane: no delta entry, no card re-enrichment from CI — DECISIONS #166(e), ruled 2026-08-29), re-verifies transform rules. Runs the harness in a fresh process per batch of 8 models with a transient per-model download (`--cache-max-bytes 0`: the 33 GB catalog fits neither the runner's disk nor the 10 GB Actions cache); a model that kills the harness even alone stays unmeasured — the measured snapshot is still committed and the job goes red naming it (DECISIONS #166). Site redeploy stays owner-dispatched while §C holds. |
| `autobump.yml` | daily · manual | custom checker for the two runtime axes: refreshes `data/releases.json`; on a new `@litertjs/core` release opens a bump PR and fires the full re-sweep. Native LiteRT release detection stays with the owner's watcher wired to Phase 8 `release-check`. Standard deps: Dependabot (`.github/dependabot.yml`). |
| `agentfix.yml` | red resweep/autobump run · manual | packages logs + delta + stale rules and invokes a headless coding agent (env-configured: `vars.AGENTFIX_COMMAND` + `secrets.AGENTFIX_API_KEY`) constrained to the failing component. Output is a **draft PR**, never auto-merged; unconfigured, it is a dispatch-testable stub that attaches the failure report only. |
| `reminders.yml` | monthly · manual | runs `freshness overdue` against `MAINTENANCE.md` and opens or bumps one reminder issue per overdue item. |

`MAINTENANCE.md` is the manual-verify register: what cannot be CI'd
(on-device card benchmarks, matrix entries needing an attached device,
unprobeable signatures), each with a re-verify interval and a
`last_verified` date the owner updates by hand.

### Sweep download cache

`npm run sweep -- … --cache-dir <dir>` downloads URL-sourced catalog models
once (content-addressed by source URL, atomic writes) and serves them from
the local server on later runs — a re-run of an unchanged catalog downloads
nothing. A failed download falls back to the direct URL so the page records
the honest `fetch_failed` result.

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

## License

Apache-2.0 (see `LICENSE`).
