---
family: silent-face
license: apache-2.0
model_id: silent-face-anti-spoofing
source_url: https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT
task: image-classification
---

# silent-face-anti-spoofing

| | |
|---|---|
| **Task** | image-classification |
| **Family** | silent-face |
| **Source** | https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python liveness/scripts/build_silentface.py` |
| **Quantization** | none — fp32 (artifact tensors: 271 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `66f0d776fb0687e376c595e729a537d6276773d84b23e5ee89317d752d95119e.tflite` | `4ff758f470b757d9005b418c9978693f09b8f2a5e7b7ee848c160db21208e1b2` | 1.765 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/silent-face-anti-spoofing.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 1.953 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 2.1784125704280783e-07 | 1.385 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/silent-face-anti-spoofing__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 2.132 | - | - | - | 135.48 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
