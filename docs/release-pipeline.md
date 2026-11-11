# Release pipeline (Phase 8): probe suite and `release-check`

`DECISIONS` and `PROGRESS` citations on this page refer to the logs of the private lab repository.

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
