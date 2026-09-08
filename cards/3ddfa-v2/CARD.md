---
family: 3ddfa
license: mit
model_id: 3ddfa-v2
source_url: https://huggingface.co/litert-community/3DDFA-V2-LiteRT
task: keypoint-detection
---

# 3ddfa-v2

| | |
|---|---|
| **Task** | keypoint-detection |
| **Family** | 3ddfa |
| **Source** | https://huggingface.co/litert-community/3DDFA-V2-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python build_tddfa.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `c916b35de9777a20a3b3bb8e11386fe6fef40be0cb1ac75d490786ddb9c8d666.tflite` | `37244f9e355d74063e6b4c5aed04d25bdf8c0bbdfe82fddbefce06cc92855579` | 6.283 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/3ddfa-v2.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 25.31 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 6.651807839577132e-05 | 1.255 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/3ddfa-v2__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 3.539 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input is a BGR (cv2-convention) face crop [1,3,120,120], normalized (x-127.5)/128; the BFM basis files are interleaved (reshape(3,-1, order='F')) — stated Kotlin-port gotchas.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
