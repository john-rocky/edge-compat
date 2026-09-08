---
family: example-family
license: apache-2.0
model_id: example-tiny-mixed
source_url: https://example.invalid/models/example-tiny-mixed
task: image-segmentation
---

# example-tiny-mixed

> **Example data.** This card carries `example`-provenance records — pipeline fixtures, not real measurements.

| | |
|---|---|
| **Task** | image-segmentation |
| **Family** | example-family |
| **Source** | https://example.invalid/models/example-tiny-mixed |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | example-converter 0.0.0-example |
| **Command** | `example-converter --input model.onnx --output model_mixed_example.tflite` |
| **Quantization** | int8-dynamic |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_mixed_example.tflite` | `9af4f555f71881924bded9d75b7e5c3febaa6c2672fbff6739cc9cf57f96f806` | 0.001 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 0.0.0-example |
| **Op coverage** | 40.0% |
| **Partitions** | 1 |
| **Blocking ops** | TRANSPOSE_CONV, FULLY_CONNECTED, CUSTOM |
| **Verdict provenance** | example=4, unmatched=1 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/examples/device_runs/0.0.0-example/2026-01-02/example-tiny-mixed__example-workstation.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| example-workstation | gpu | pass | - | - | 256 | - | 123.4 | 45.6 | 78.9 | 512.0 | Example Workstation · Example Desktop SoC · litert-lm 0.0.0-example · macOS 0.0-example | 2026-01-02 | example |

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
