# edge-card: perf cards

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
- **Zero matrix writes, ever** (the trap rule — see [Browser backends](browser.md#browser-backends-phase-6));
  covered by a test.
- Exit codes: `0` written · `1` invalid sweep result or card, or a device cell
  a card carries would be dropped without `--allow-drop` (nothing written)
  · `2` usage error.

Index browser-status vocabulary (a summary cell — the evidence stays in the
card and the linked sweep record): `load_failed` / `run_failed` /
`output_mismatch` (ran, numerics outside tolerance vs the wasm_xnnpack
reference) / `fallback` (ran, but not fully delegated) / `pass` (loaded, ran,
and neither a mismatch nor a fallback was recorded).
