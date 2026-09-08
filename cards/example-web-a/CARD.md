---
family: example-family
license: apache-2.0
model_id: example-web-a
source_url: https://example.invalid/models/example-web-a
task: example-elementwise
---

# example-web-a

> **Example data.** This card carries `example`-provenance records — pipeline fixtures, not real measurements.

| | |
|---|---|
| **Task** | example-elementwise |
| **Family** | example-family |
| **Source** | https://example.invalid/models/example-web-a |
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
| `model_web_a_example.tflite` | `2e852fc9cad7901357562f055399694f8292d43fa780d7d5fe33cfa8f5e1372a` | 0.0 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/examples/sweep/example-web-a.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 0.032 | example-dev-mac-arm64 · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-10 | example |
| webgpu_mldrift | pass | yes | pass | 1.3937262805208635e-07 | 0.55 | example-dev-mac-arm64 · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-10 | example |

## Pitfalls

- Example card for the browser-sweep pipeline — synthetic fixture, not a real model.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
