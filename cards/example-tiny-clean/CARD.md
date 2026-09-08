---
family: example-family
license: apache-2.0
model_id: example-tiny-clean
source_url: https://example.invalid/models/example-tiny-clean
task: image-classification
---

# example-tiny-clean

> **Example data.** This card carries `example`-provenance records — pipeline fixtures, not real measurements.

| | |
|---|---|
| **Task** | image-classification |
| **Family** | example-family |
| **Source** | https://example.invalid/models/example-tiny-clean |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | example-converter 0.0.0-example |
| **Command** | `example-converter --input model.onnx --output model_clean_example.tflite` |
| **Quantization** | none |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_clean_example.tflite` | `e763326d958dca53e16643c11ce300fbac480c475cfd751b6d8336650089670f` | 0.001 |

## Performance

| Device | SoC | Backend | Precision | Latency p50 (ms) | Tokens/s | Peak mem (MB) | Harness | Date | Provenance | Additional metrics |
|---|---|---|---|---|---|---|---|---|---|---|
| Example Device | Example SoC | gpu_mldrift | f32 | 1.2 | - | 12.5 | example-harness 0.0.0-example (median of 10) | 2026-08-10 | example | decode_ms=0.7 · encode_ms=0.5 · rtf=0.01 |
| Example Device | Example SoC | cpu_xnnpack | f32 | 4.8 | - | 9.0 | example-harness 0.0.0-example (median of 10) | 2026-08-10 | example | - |

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 0.0.0-example |
| **Op coverage** | 100.0% |
| **Partitions** | 1 |
| **Blocking ops** | none |
| **Verdict provenance** | example=3 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/examples/device_runs/0.0.0-example/2026-01-01/example-tiny-clean__example-phone.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| example-phone | cpu_xnnpack | pass | - | - | - | 4.8 | - | - | - | 9.0 | Example Phone · Example SoC · litert 0.0.0-example · EXAMPLE.240001.001 | 2026-01-01 | example |
| example-phone | npu_qnn | fallback | no | pass | - | 1.2 | - | - | - | - | Example Phone · Example SoC · litert 0.0.0-example · QNN SDK 0.0.0-example · EXAMPLE.240001.001 | 2026-01-01 | example |

## Cross-runtime

| Runtime | Device | SoC | Backend | Precision | Latency p50 (ms) | Tokens/s | Peak mem (MB) | Harness | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| example-other-runtime | Example Device | Example SoC | cpu_xnnpack | f32 | 5.1 | - | 11.0 | example-harness 0.0.0-example (median of 10) | 2026-08-10 | example |

## Pitfalls

- Example pitfall — this entire card is a pipeline fixture, not a real model.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
