---
family: rtmpose
license: apache-2.0
model_id: rtmw-m-wholebody
source_url: https://huggingface.co/litert-community/RTMW-m-WholeBody-LiteRT
task: keypoint-detection
---

# rtmw-m-wholebody

| | |
|---|---|
| **Task** | keypoint-detection |
| **Family** | rtmpose |
| **Source** | https://huggingface.co/litert-community/RTMW-m-WholeBody-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python rtmw/scripts/build_rtmw.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `22284ea4a7d56f6a62c635ee21e0c4aa5b089d65c4da8050312ec6b2eda2d29a.tflite` | `66f79625d2054dfc5021fa22b38f0b76a1d5f7111cc790d5d9b65ee04a5985d6` | 63.073 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/rtmw-m-wholebody.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 561.783 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.06526136062367838 | 13.322 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/rtmw-m-wholebody__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/rtmw-m-wholebody__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 11.882 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 2.659 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 109.617 | - | - | - | 273.34 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
