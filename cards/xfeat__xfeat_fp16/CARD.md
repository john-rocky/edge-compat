---
family: xfeat
license: apache-2.0
model_id: xfeat__xfeat_fp16
source_url: https://huggingface.co/litert-community/xfeat-litert
task: keypoint-detection
---

# xfeat__xfeat_fp16

| | |
|---|---|
| **Task** | keypoint-detection |
| **Family** | xfeat |
| **Source** | https://huggingface.co/litert-community/xfeat-litert |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python xfeat/scripts/convert_xfeat.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `20d17fe96505e37a7966e8e038fcbdcb6199340d23c8a6b1aaf759cfc4b01014.tflite` | `b3ff3c12e5bad9bde8c0a421a1bd18ceca35b6a1c01a51dcac79f2614df37a0f` | 1.349 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/xfeat__xfeat_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 189.197 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.1568627450980392 | 12.378 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/xfeat__xfeat_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/xfeat__xfeat_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 4.313 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 384.541 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 32.452 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Per-image InstanceNorm is applied host-side ((g-mean)/sqrt(var+1e-5) over the grayscale image) — its spatial reduction would overflow fp16 on the delegate (HF card I/O + re-authoring sections).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
