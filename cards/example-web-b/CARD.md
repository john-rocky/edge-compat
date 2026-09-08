---
family: example-family
license: apache-2.0
model_id: example-web-b
source_url: https://example.invalid/models/example-web-b
task: example-elementwise
---

# example-web-b

> **Example data.** This card carries `example`-provenance records — pipeline fixtures, not real measurements.

| | |
|---|---|
| **Task** | example-elementwise |
| **Family** | example-family |
| **Source** | https://example.invalid/models/example-web-b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert_compat.parser.fixtures 0.1.0 |
| **Command** | `python -m litert_compat.parser.fixtures` |
| **Quantization** | none |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_web_b_example.tflite` | `5836dace323eb386f01224c8aeaa3b218dfd499d067e61437c250e198aaf489d` | 0.001 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/examples/sweep/example-web-b.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 0.043 | example-dev-mac-arm64 · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-10 | example |
| webgpu_mldrift | pass | yes | pass | 6.83038714341598e-07 | 0.635 | example-dev-mac-arm64 · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-10 | example |

## Pitfalls

- Example card for the browser-sweep pipeline — synthetic fixture, not a real model.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
